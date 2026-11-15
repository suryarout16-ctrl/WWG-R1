# 仮想R1で発話と動作を組む

以下は `local_tools/virtual_r1.py` で演出の順序を検討した記録。現在、仮想R1は凍結しており、起動・接続手順は再開指示まで実行しない。現行の実機・ローカル試験は [即興デモ](LIVE_DEMO.md) を参照。
この仮想器はDDSの応答と状態遷移をまねるもの。歩行の安定性、衝突、関節軌道、正確な距離は検証しない。

## 最初の仮台本

`scenarios/training_rehearsal.json` は、具体的な作業指示書を受け取る前のリハーサル用。

| 場面 | 発話と動作の組み合わせ |
|---|---|
| 自己紹介 | 固定の自己紹介と手振りを重ねる |
| 指示書へ導入 | 指示書を一緒に確認する声かけと強調動作 |
| 位置を変える演出 | 発話と前進指令を重ね、2秒後にゼロ速度。完了を待って次へ |
| 停止の区切り | ゼロ速度と次の確認への声かけ |
| 新人役の模擬ミス | 「確認せずに進めてよいか」という発言 |
| いったん止める | 冗談・身ぶりを入れず、確認を優先 |
| 振り返りへ切り替え | 進行役が危険解消済みの模擬設定を明示 |
| ヒヤリハット | 台詞と帽子の演出予約。専用動作は未実装として送信しない |
| 励ます | 質問を歓迎する声かけと胸に手を当てる動作 |

腕の動作・保持の解除を待ち、歩行と腕の動作を重ねない。発話はその場面の動作と重ねられる。
正式な台本・架空の作業指示書が届いたら、場面の順序・台詞・新人役の発言を差し替える。教材にない機械操作の指定値は作らない。

## プレビュー

```powershell
pixi run python scripts/virtual_scenario.py scenarios/training_rehearsal.json
```

既定は順序の表示だけで、DDS通信も音声再生も行わない。自己紹介は `r1_chat/identity.py` の固定文を参照する。

## 仮想器へ接続する

仮想器本体と録画用ツールはコミット `42a1d77` に含まれる。録画データはGitに含めていないため、別PCでは `logs/r1_snapshot_20261114_172704/` フォルダを丸ごと同じ相対位置へコピーする。
まず別のターミナルで、ループバック上のRunning（811）から起動する。既に仮想器が動いている場合は重ねて起動しない。

```powershell
pixi run python local_tools/virtual_r1.py --fsm 811 --mute --snapshot logs/r1_snapshot_20261114_172704
```

以下のシナリオランナー・台本と依存コードはこの版に含まれる。別PCではGitに含めない録画フォルダとPC固有設定を別途用意する。

```powershell
pixi run python scripts/virtual_scenario.py scenarios/training_rehearsal.json --run-virtual
# PCから台詞も聞く場合
pixi run python scripts/virtual_scenario.py scenarios/training_rehearsal.json --run-virtual --pc-audio
```

実行先は `Loopback Pseudo-Interface 1` に固定し、Ethernetを指定する引数は設けない。
ログは `logs/` に仮想リハーサルとして保存する。仮想器の応答127や動作完了を実機の成功として記録しない。
帽子は未実装の演出予約として表示するだけで、任意の教示動作を呼び出さない。

現在の仮想器と通常パネルの監視方式の違いは [互換性メモ](VIRTUAL_R1_COMPATIBILITY.md) を参照。

## 今回の確認

2026-11-14に、上記9場面を `--run-virtual --pc-audio` で通し実行し、完了した。
PC音声再生の経路、仮想の腕動作・保持解除、前進後のゼロ速度、場面間の待ち合わせを実行した。新人役・進行役は表示のみ、帽子は未実装の演出予約として送信していない。
記録は `logs/virtual_scenario_20261114_231933_7b8776.json`。`mode: virtual_only`、`physical_result: not_tested` として実機の記録と区別する。
通常パネルのCRC・鮮度監視や自由会話のStrataを使う試験とは別で、固定の仮台本と仮想サービスの組み合わせを確認したもの。
