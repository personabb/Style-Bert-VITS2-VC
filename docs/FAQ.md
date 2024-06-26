# よくある質問

## Google Colabでの学習がエラーが何か出て動かない

Google Colabのノートブックは以前のバージョンのノートブックのコピーを使っていませんか？
Colabノートブックは最新のバージョンに合ったノートブックで動かすことを前提としています。ノートブック記載のバージョンを確認して、[最新のcolabノートブック](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)（を必要ならコピーして）から使うようにしてください。

## 学習に時間がかかりすぎる

デフォルトの100エポックは音声データ量によっては過剰な場合があります。デフォルトでは1000ステップごとにモデルが保存されるはずなので、途中で学習を中断してみて途中のもので試してみてもいいでしょう。

またバッチサイズが大き過ぎてメモリがVRAMから溢れると非常に遅くなることがあります。VRAM使用量がギリギリだったり物理メモリに溢れている場合はバッチサイズを小さくしてみてください。

## どのくらいの音声データが必要なの？

分かりません。試行錯誤してください。

参考として、数分程度でも学習はできるらしく、またRVCでよく言われているのは多くても45分くらいで十分説があります。ただ多ければ多いほど精度が上がる可能性もありますが、分かりません。
<!-- OpenJTalkの間違ったトーンで大量のデータを学習すると間違ったトーンの入力でなければ正しい出力ができなくなるが、学習データの範囲内ならば高い性能を発揮する -->

## どのくらいのステップ・エポックがいいの？

分かりません。試行錯誤してください。`python speech_mos.py -m <モデル名>`によって自然性の一つの評価ができるので、それが少し参考になります（ただあくまで一つの指標です）。

参考として、最初の2k-3kで声音はかなり似始めて、5k-10k-15kステップほどで感情含めてよい感じになりやすく、そこからどんどん回して20kなり30kなり50kなり100kなりでどんどん微妙に変わっていきます。が、微妙に変わるので、どこがいいとかは分かりません。

## APIサーバーで長い文章が合成できない

デフォルトで`server_fastapi.py`の入力文字上限は100文字に設定されています。
`config.yml`の`server.limit`の100を好きな数字に変更してください。
上限をなくしたい方は`server.limit`を-1に設定してください。

## 学習を中断・再開するには

- 学習を中断するには、学習の進捗が表示されている画面（bat使用ならコマンドプロンプト）を好きなタイミングで閉じてください。
- 学習を再開するには、WebUIでモデル名を再開したいモデルと同じ名前に設定して、前処理等はせずに一番下の「学習を開始する」ボタンを押してください（「スタイルファイルの生成をスキップする」にチェックを入れるのをおすすめします）。

## 途中でバッチサイズやエポック数を変更したい

`Data/{モデル名}/config.json`を手動で変更してから、学習を再開してください。

## その他

ググったり調べたりChatGPTに聞くか、それでも分からない場合・または手順通りやってもエラーが出る等明らかに不具合やバグと思われる場合は、GitHubの[Issue](https://github.com/litagin02/Style-Bert-VITS2/issues)に投稿してください。
