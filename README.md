#<BR>
# 英語のテキストが埋め込まれた論文や書籍等のpdfファイル（複数可）をOpenAIのAPIを使用して日本語に翻訳するアプリ（並列処理）<BR>
# Ver.1.0<BR>

#<BR>
#論文等をフォルダに入れて一括で日本語に翻訳します。<BR>
#OpenAIのAPIキーが必要です。<BR>
#並列処理で動かしますが、接続エラーには対応していません。その場合翻訳されずに終わります。<BR>
#ある程度の長さの書籍や論文は分割して翻訳します。<BR>
#<BR>
#書籍であれば1冊あたり0.2-0.5ドルほどかかります（長さによる）。<BR>
#論文は5本ほどで0.02ドルほどです（長さによる）。<BR>
#<BR>
#gpt-4o-mini-2024-07-18を使用しています（gptの2024年のモデルの廉価版）。<BR>
#<BR>
