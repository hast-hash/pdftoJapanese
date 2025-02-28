# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:00:43 2025

@author: hast
"""
#
# 英語のテキストが埋め込まれた論文や書籍等のpdfファイル（複数可）をOpenAIのAPIを使用して日本語に翻訳するアプリ（並列処理）
#

#pdfが保存されているフォルダ
dir=""
#保存用フォルダ
dir2=""

th=2000 #サイズの大きいテキストを切り分けるときの目安のワード数（英語など）

#API_KEY for local PC
import os
API_KEY = os.environ["OPENAI_API_KEY"] #ローカルPCの環境変数にOpenAIのAPIキーを設定
#API_KEY for Google Colab
#from google.colab import userdata
#API_KEY = userdata.get('OPENAI_API_KEY')

##########################
##########################
##########################

model_id = 1 #モデルとプロンプトはここで設定
instruct_prompt_dataset=[
    [""],
    ["gpt-4o-mini-2024-07-18", "You are a translater who translate English texts into Japanese. You will be provided with an English text. Please translate the English text into Japanese."],

]

##########################
##########################
##########################

# ライブラリインポート
#!pip install tiktoken
#!pip install pdfminer.six
#!pip install tenacity
#!pip install Spire.PDF
#!pip install backoff
#!pip3 install pycopy-collections-abc
#!pip install nest_asyncio
#!pip install pdfplumber

#
#OpenAIを使ってファイルを並列に翻訳
#
import pandas as pd
import time
import glob
from pathlib import Path
import nest_asyncio
nest_asyncio.apply()
#import numpy as np
import os
#import re
#from openai import OpenAI
#import json

from openai import AsyncOpenAI
import asyncio

from spire.pdf import PdfDocument
from spire.pdf import PdfTextExtractOptions
from spire.pdf import PdfTextExtractor
import pdfplumber

from more_itertools import chunked
from tqdm import tqdm
from typing import TypedDict
from collections.abc import AsyncGenerator, Generator, Iterable
#import argparse
#from custom_types import Example
#import datetime
#from pdfminer.high_level import extract_text
#from typing import AsyncIterator
#import pdfminer
#from tenacity import retry, stop_after_attempt, wait_random_exponential
#import backoff

#client = OpenAI(api_key=API_KEY)
client = AsyncOpenAI(api_key=API_KEY)

instruct_prompt = instruct_prompt_dataset[model_id][1]
model_name = instruct_prompt_dataset[model_id][0]

finetune_switch=False #Trueの場合ファインチューン済みモデルを使用する

#ディレクトリの確認
if not os.path.isdir(dir):
    os.makedirs(dir)
if not os.path.isdir(dir2):
    os.makedirs(dir2)
    
#pdfからテキストを抽出 pdfDocumentを使用 -> 短い論文などではOK
spirepdftext="Evaluation Warning : The document was created with Spire.PDF for Python."
def extractpdftext(pdffile):
    pdf = PdfDocument()
    pdf.LoadFromFile(pdffile)
    extracted_text = ""
    extract_options = PdfTextExtractOptions()
    for i in range(pdf.Pages.Count):
        page = pdf.Pages.get_Item(i)
        text_extractor = PdfTextExtractor(page)
        text = text_extractor.ExtractText(extract_options)
        extracted_text += text
    extracted_text=extracted_text.replace(spirepdftext,"")
    return extracted_text

#書籍など長めのテキストはpdfplumberを使用
def extractpdftext2(file):
    all_text=""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text
            
# PDFファイルからテキストを抽出
df=pd.DataFrame()
text=""
#laparm = pdfminer.layout.LAParams()
#laparm.word_margin = 0.01
print("Extracting texts from pdf files...")
for file in glob.glob(dir+'/**/*.pdf', recursive=True):
    source = Path(file)
    print(source.name)
    #text = extract_text(source) #pdfDocument使用時
    text=extractpdftext2(file) #pdfplumber使用時
    textwords=text
    textwords=len(textwords.split()) #大雑把なワード数
    if textwords<th: #サイズの小さいテキストはそのまま翻訳
        df2=pd.DataFrame({'text': [text],
                    'file': [source.name],
                    'total': 1,
                    'num': 1})
    else: #テキストが大きい場合小さく切り分け
        text2=text.split('\n')
        tpn=len(text2) #段落数
        ts=-(-textwords//th) #総切り分け数
        pn=-(-tpn//ts) #一つの切り分けの段落数
        df2=pd.DataFrame()
        for i in range(ts):
            if pn*(i+1)>tpn: 
                text3="\n".join(text2[pn*i:])
            else:
                text3="\n".join(text2[pn*i:pn*(i+1)])
            df3=pd.DataFrame({'text': [text3],
                        'file': [source.name],
                        'total': ts,
                        'num': i+1})
            df2 = pd.concat([df2,df3])
    df = pd.concat([df,df2])
df = df.reset_index() #dfにすべてのテキストが入る

class OpenAIResponse(TypedDict):
    response: str
    
async def call_api(
    prompts: Iterable[str],
    chunk_size: int,
    model_name: str = "gpt-4o-mini-2024-07-18",
    temperature: float = 0.0,
) -> list[OpenAIResponse]:
    responses: list[OpenAIResponse] = []
    k=0
    for chunk in tqdm(chunked(prompts, chunk_size)):
        coroutines = [client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": instruct_prompt},
                    {"role": "user", "content": prompt}
                    ],
                temperature=temperature,
                max_tokens=16384,
            )
            for prompt in chunk
            ]
        print('Processing...')
        print(df[['file','total','num']][k*chunk_size:(k+1)*chunk_size])
        k+=1
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        df5=pd.DataFrame(results)
        for p in range(len(df5)):
            if isinstance(p, Exception):
                mes="response None"
                responses.append([mes])
                print(mes,p)
            else:
                tr=df5[1][p][1][0].message.content
                tr=tr.strip()
                responses.append(
                        [tr]
                )
    return responses

async def main(examples, chunk_size: int):
    responses = await call_api(examples, chunk_size)     
    return responses

# 質問のリスト作成
questions = "TEXT: "+df["text"]
detail_prompts=questions.tolist()

time_start = time.time() 
chunk_size=5
responses=asyncio.run(main(detail_prompts, chunk_size))
time_end = time.time()
print(time_end-time_start)

df6=pd.DataFrame(responses,columns=["translation"])
df7=pd.concat([df6,df],axis=1)
#now = datetime.datetime.now()
#df7.to_csv(dir+"translation_"+now.strftime('%Y%m%d_%H%M%S')+".csv", sep='\t')

#結合
df8 = (df7.groupby(['file'])['translation']
          .apply(list)
#          .apply(lambda x:sorted(x)) #ソートはしない
          .apply('\n'.join)
         )

#保存 pdfと同じ名前で拡張子をtxtにして保存
writetext=""
for i in range(len(df8)):
    file2=Path(df8.index[i]).stem
    file2=dir2+file2+'.txt'
    df9=df8[i]
    with open(file2, mode="a", encoding='utf-8') as f:
        f.write(df9)

#df.to_csv(file, index=False, encoding='utf-8')
