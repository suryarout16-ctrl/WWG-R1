# WWG-R1：Unitree R1 日本語チャットボット

Unitree R1 を、日本語で会話する受付・案内ロボットにする。
プログラムはすべて手元の PC で動かし、LAN（DDS）経由で R1 に命令を送る。R1 の Onboard PC には触らない。

**当日の運用は [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md)（デモ運用手順書）を見る。**

## 技術的アプローチ

```
PCマイク ──▶ faster-whisper (small / ja / VAD) ──▶ OpenAI API (Responses) ──▶ Windows SAPI「Microsoft Ichiro」(男性)
                                                                                 │ 16kHz mono PCM
                                                                                 ▼
                                                     unitree_sdk2py AudioClient.PlayStream ──DDS──▶ R1 頭部スピーカー
```

1. **音声入力**：PC マイクで録音し、faster-whisper（small、GPU が使えなければ CPU int8 に自動切替）で文字にする
2. **対話生成**：OpenAI Responses API。会話履歴を保持し、読み上げ向けに 1〜2 文の短い返答にする
3. **音声合成**：R1 標準の TtsMaker は英語・中国語のみなので、日本語は PC 側で合成する（Windows 標準の男性音声 Ichiro、オフライン）
4. **再生**：AudioClient.PlayStream で 1 秒ずつ PCM を送り、R1 のスピーカーから流す（R1 専用の音声クライアントは SDK に無いので G1 用を使う）
5. **安全・安定性**：体は動かさない（音声のみ）。通信断時は定型文で応答して止まらない。`--say` → `--echo` → 本番の順で段階的に確認する

## 使用リポジトリ・SDK

| 名前 | 用途 |
|---|---|
| [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | R1 との通信、音声再生（AudioClient） |
| CycloneDDS 0.10.2 | PC と R1 の DDS 通信（R1 側と同じ版。11.x では通信できない） |
| [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 日本語音声認識 |
| OpenAI Python SDK | 対話生成 |
| Windows System.Speech（Microsoft Ichiro） | 日本語音声合成 |
| sounddevice / NumPy / SciPy | マイク入力、音声変換 |
| 参考：[GAI-313/humanoid_hackathon_env](https://github.com/GAI-313/humanoid_hackathon_env)、[syswonder/robot-unitree-r1](https://github.com/syswonder/robot-unitree-r1) | ハッカソン環境、R1 音声インターフェースの実装例 |

## セットアップ

1. Git と [pixi](https://pixi.prefix.dev/latest/installation/) を入れる
2. `git clone https://github.com/suryarout16-ctrl/WWG-R1.git` → `cd WWG-R1` → `pixi install`
3. `pixi run verify-sdk` で `unitree_sdk2py & cyclonedds loaded successfully!` が出れば OK
4. OpenAI の API キーを、このPCの中だけに保存する（リポジトリやチャットには書かない）
   Windows: `pixi run python r1_chat/credentials.py set`（資格情報マネージャーに暗号化保存。`status` で確認）
   その他の OS: 環境変数 `OPENAI_API_KEY`
   PowerShell: `[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "<キー>", "User")` → 開き直す
5. （任意・Windows で GPU 認識）cuBLAS 12 / cuDNN 9 の DLL があるフォルダを `R1_CUDA_DLL_DIR` に入れる。無ければ CPU で動く
6. Windows は有線 LAN アダプタ名を英数字（例 `Ethernet`）に変えておく

## R1 との接続

LAN ケーブルで R1 のうなじのポートと PC をつないだら、1 本のスクリプトで設定〜確認まで行う（Windows、管理者権限は自動で求められる）:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_lan.ps1
```

1. 有線 LAN アダプタを探し、名前を `Ethernet` に変える
2. 固定 IP `192.168.123.164/24`（ゲートウェイなし。ほかの IPv4 は外し、元の設定は `.lan_backup.json` に保存）
3. ネットワークを「プライベート」に（「パブリック」だと 3102 エラーの原因になる）
4. `192.168.123.161` に ping
5. R1 に日本語で一言話させる（`pixi run chat -i Ethernet --say ...`）

- `-DryRun` で何も変えずにやることだけ表示、`-Restore` で元に戻す
- 実行内容と最後のアダプタ状態は `logs\setup_lan_*.log` に残る

## 使い方（リポジトリのルートで）

| やりたいこと | コマンド |
|---|---|
| R1 に一言話させる（LLM 不要） | `pixi run chat -i Ethernet --say "こんにちは、僕はR1です"` |
| マイク→認識→声 の確認（おうむ返し、LLM 不要） | `pixi run chat -i Ethernet --echo` |
| 本番（声で会話） | `pixi run chat -i Ethernet` |
| R1 なしで PC のスピーカーで試す | `pixi run chat --local` |
| マイクなし（キーボード） | `pixi run chat --local --text` |

会話中：文字を打って Enter → 送信 / 何も打たずに Enter → 録音開始、もう一度 Enter で停止 / `q` で終了

### ログ

`logs/` に実行ごとに残る（Git には含めない）:
- `r1_chat_<日時>_<chat|echo|say>.log`：画面と同じ内容（時刻付き、エラーの詳細も）
- `r1_chat_<日時>_<モード>_turns.jsonl`：1 往復ごとの記録（入力、認識結果、返答、認識・LLM・合成の秒数、エラー）
- `--save-audio` で録音した声も `logs/audio/` に WAV で残す。`--no-log` でログなし

主なオプション：`--volume 70`（R1 の音量、上限 100）、`--gain 9`（さらに音圧を上げる dB）、`--rate 2`（Ichiro の速さ -10〜10）、`--model gpt-5-mini`（または環境変数 `OPENAI_MODEL`）、
`--tts openai --voice onyx`、`--mic-device <番号>`、`--stt-device cpu --stt-compute-type int8`

## 安全ルール（ハッカソン運用ガイドより）

- Onboard PC にログインしない。プログラムは PC から
- 初めて動かすプログラムは R1 を吊った状態で。床では周囲 2 m 以上空ける
- 動作コマンドを送るときは、コントローラー（E-Stop）を持つオペレーターを別に置く
- 日本語の音声を流すことは事前にスタッフに相談する

## 確認状況（2026-11-14）

- 確認済み：Ichiro 合成（2 回目以降 0.1〜0.7 秒）、faster-whisper small（GPU 約 1.2 秒 / CPU 約 2.2 秒）、PC マイク録音、`--local --text --echo` の会話ループ
- 未確認：OpenAI 応答、R1 実機での PlayStream 再生
- できないこと：脚の直接制御、SLAM（LiDAR なし）、正確な距離移動、把持。詳細は [docs/R1_guide.md](docs/R1_guide.md)
