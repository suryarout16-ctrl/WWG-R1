# 技術Q&A 参照資料（研修担当R1）

研修担当R1が「仕組み」を聞かれたときに話す内容の根拠です。R1が実際に参照するのは
`r1_chat/tech_facts.py` の `TECH_FACTS`（短い要約）で、仕組みの質問だと判定したときだけ指示文に追加します。
この資料は人間向けに、各事実がコードのどこで決まっているかを示します。**コードを変えたら、この資料と `tech_facts.py` の両方を直してください。**

## 会話から発話・動作までの流れ

```
マイク(PC) ─録音→ faster-whisper(PC, small) ─文字→ LocalBrain ─指示文+履歴→ Strata(Z840) / GPT(手動切替)
                                                     │
                       ┌──── JSON {say, gesture} ←───┘
                       ▼
              検査（形式・長さ・許可された身ぶり）── 不合格 → 身ぶりなしの文で1回だけ作り直し
                       │
          say ─→ Windows SAPI 音声(PC) ─16kHz PCM, 1秒ずつ→ DDS voice PlayStream → R1のスピーカー
          gesture ─→ 腕: Arm Action ID（7106）  /  歩行: sport 7105（利用者の直接依頼のときだけ）
```

設計思想: **AIは何をするか決める。プログラムはどう実現するか決める。安全機構は実行してよいか決める。**
（`r1_chat/identity.py` の `MentorIdentity.principle`）

## 会話の三つの領域（本番は LLM の即興対話）

本番は台本の読み上げではなく、LLM が同じ人格のまま `say` と `gesture` を同時に作ります（`personas.DOMAIN_RULE`）。

| 領域 | 例 | LLM に許すこと | 根拠 |
|---|---|---|---|
| 1. 架空の作業 | 「この機械で昔どんな失敗が？」「How does this imaginary machine move boxes?」 | 設定・作業例・昔話・冗談を創作してよい。教材の合意事項（A=移動、B=停止、C=押し間違い用）と会話で決めた設定は守る。毎回「架空です」と断らない | 世界設定資料 `scenarios/live_training_context.txt`（Codex 作成） |
| 2. R1自身の本当の仕組み | 「どういう仕組みで話してるの？」「実際のR1の関節トルクは？」 | この資料の事実だけ。載っていない値・機能は「確認できていない」「分からない」 | `tech_facts.TECH_FACTS`（仕組みの質問と判定したときだけ指示文に追加） |
| 3. 存在意義 | 「君はなんのためにここに？」「自己紹介して」 | 意味（新人に機械操作を教え、操作ミスによる事故を防ぐ／AI・プログラム・安全確認の役割分担）を保ち、言い回しは自由 | `identity.SELF_DESCRIPTION_RULE` |

- 仕組みの質問かどうかは `tech_facts.is_technical_question` が判定する。Strata・LoRA・GitHub などシステム固有の語があれば常に領域2。「仕組み」「モーター」などの話題語は、R1自身（あなた・R1・実際の…）を指すときは領域2、架空の機械だけを指すときは領域1（創作に任せる）。
- どの領域でも、R1の実機の機能・関節の値・動作の実行結果・周囲の認識は作らせない（`DOMAIN_RULE` の最終行）。
- 移動を選んだ返事は LLM の言葉のまま話す。ただし「動いた」「着いた」「安全を確認した」「1メートル」など実行結果・周囲の認識・距離を語った文だけ、プログラムの決まった文（`BODY_ANNOUNCEMENTS(_EN)`）に置き換える（`personas.motion_claim`）。
- 自己紹介の固定文（`identity.SELF_INTRODUCTION(_EN)`）は回帰試験用。`LocalBrain.fixed_lines = True` のときだけ使う。本番では、移動の結果を語った自己紹介を差し替えるときだけ使う。
- 「ハルシネーションゼロを保証」は実装していない。事実集の提示・分からないときの答え方・評価（下記）で抑える。

## 実装の版と確認の段階

この版には、Strata/GPT切替、自己像と会話の三領域、研修資料、歩行指令と状態監視、日英対応、技術の事実集、発表シナリオ、実機・ローカル起動が含まれる。APIキー・SSH接続情報と鍵・PC固有設定・会話ログ・録画データはGitに含めない。

以下の実装欄はこの資料と同じ版を指す。実行中のコード、起動済みプロセス、GitHub上の最新ブランチが同じとは限らない。具体的な公開状況を問われたらコミットと差分を確認し、未確認のまま未公開・公開済みと断定しない。コードが含まれることと、実機で確認済みであることも区別する。

確認の段階: **実機** = R1実機で動作確認 / **GPT1回** = Codex が有料APIで1回確認 / **仮想** = 仮想R1・シナリオで確認 / **単体** = 単体テストのみ / **未** = 未確認

## 事実と根拠

| 話題 | 事実 | 根拠（コード） | 実装の版 | 確認 |
|---|---|---|---|---|
| 音声認識 | PCで faster-whisper（既定 small）。別プロセス `stt_worker.py` で動く | `r1_chat/r1_chat.py` `WhisperSTT`、`r1_chat/stt_worker.py` | 本版 | 実機（日本語） |
| 日英の聞き分け | `--stt-language auto`（既定）で日本語と英語だけを候補に言語判定。日本語優先で、英語の確率が 0.65 以上なら英語で文字にする。ヒンディー語などへの誤判定を防ぐため候補を2つに絞る。日本語と判定しても、文字がほぼカタカナだけ（なまりの強い英語の特徴）なら英語でもう一度文字にする | `stt_worker.py` `parse_languages` / `choose_language` / `looks_like_katakana_english`、`web_ui.py` `--stt-language` | 本版 | 単体＋合成音声8件（下記） |
| 認識の確信度 | 平均対数確率が `--stt-low-logprob`（既定 -1.0）未満、日英判定が 0.6 未満、またはカタカナから英語へ文字にし直したときは「確信度低め」とし、機械操作の話なら推測せず聞き返す指示を追加 | `web_ui.py` `Engine._low_confidence`、`bilingual.LOW_CONFIDENCE_RULE` | 本版 | 単体 |
| 返答を作るAI | 通常構成は Z840 上の Strata（`Qwen3.8-Flash-Next` の IQ3_XXS 量子化、約3ビット）。思考モードは無効（`enable_thinking: false`, `reasoning_effort: none`）。PCからSSH中継で呼ぶ。現在の応答先とモデルは各Brainの `BACKEND_LABEL` / `model` と稼働状態から毎回与え、通常構成より優先する。GPT使用中にStrataが今の返答を作ったとは説明しない | `personas.py` `LocalBrain._messages` / `_stream`、`mentor_backend.py`、`r1_chat/ssh_relay.py` | 本版 | 実機（Strata）・単体（応答先の指示） |
| 予備のAI | OpenAI GPT API（既定 `gpt-5-mini`）。担当者がパネルで手動で切り替える。自動切替はない | `r1_chat/mentor_backend.py` `OpenAIMentorBrain`、`web_ui.py` | 本版 | GPT1回 |
| 待ち時間 | Strata の待ち時間は既定10秒（`--strata-timeout`）。超えたら通信エラーの定型文を話す。11/14夜は同じ質問が10.01秒でタイムアウトした | `web_ui.py` `_strata_timeout`、`bilingual.ERROR_REPLY` | 本版 | 実測 |
| AIの出力形式 | `{"say": 文, "gesture": 名前}` のJSONだけ。gesture は許可一覧の enum。文の長さ上限は日本語240字・英語400字 | `personas.py` `LocalBrain.reply`、`MAX_SAY_CHARS(_EN)` | 本版 | 実機 |
| 検査と作り直し | 形式・長さ・`<think>` 混入・許可外の身ぶりを検査し、だめなら身ぶりなしの平文で1回だけ作り直す | `personas.py` `_structured_reply` / `_speech` | 本版 | 単体 |
| 移動前の発話 | AIの文のまま話す。実行結果・周囲の認識・距離を語った文だけプログラムの決まった文に置き換える。自己紹介は AI の言葉（固定文は回帰試験用） | `personas.py` `motion_claim` / `BODY_ANNOUNCEMENTS`、`bilingual.BODY_ANNOUNCEMENTS_EN`、`LocalBrain.fixed_lines` | 本版 | 単体 |
| 話す | Windows SAPI で合成（日本語 Microsoft Ichiro、英語 Microsoft David。どちらも男性の声）。PowerShell 7（pwsh）で動かし、指定の声を選べなければ起動を止める。16kHz・モノラルのPCMを1秒（32000バイト）ずつ DDS の voice サービス `PlayStream` で送る | `r1_chat.py` `SapiTTS`、`SAMPLE_RATE`、`CHUNK_BYTES`、`web_ui.py` `Engine._tts_for` | 本版 | 実機でPlayStream再生済み。**ただし11/15の修正まで、Windows PowerShell 5.1 が Ichiro を選べず既定の Haruka（女性）で話していた**。修正後の Ichiro / David は PC上の合成のみ確認、実機は未 |
| 腕の身ぶり | Unitree Arm Action の固定ID: 手を振る25、大きく手を振る26、拍手17、ガッツポーズ29、胸に手33、強調35、前方を示す31。姿勢を保つ動作（29/31/33/35）は話し終わったらID 99で戻す。立って動ける状態（FSM 811/816）のときだけ送る | `r1_chat.py` `GESTURES` / `HOLD_IDS` / `STANDING_FSM` | 本版 | 実機（31の保持は未） |
| 帽子の身ぶり | 教示動作の予約名 `hat_tip` があるだけで、専用動作は未実装 | `r1_chat.py` `GESTURES`、`r1_motion/hat_gesture.json` | 本版 | 未 |
| 会話からの単発移動 | sport API 7105 の速度指令。1回最大2秒、前後左右 0.3 m/s 以下、旋回 0.5 rad/s 以下、最後に速度ゼロ。Running(811) が必要。この機体で観測した127は物理的な成功を意味しない。脚の関節データで変化・静止を観測するが、実際の移動・停止には目視との照合が必要。127の場合は `physical_confirmation_required` が残る。距離や到達は保証しない | `r1_motion/body_control.py`（`max_duration`、上限検査、`_Call(7105, …)`、`state`）、`r1_motion/body_feedback.py` | 本版 | 実機（0.3 m/s×2秒の前進と停止を利用者が目視確認） |
| 歩行の許可 | AIが移動を選んでも、利用者が短い命令文で直接頼んだときだけ実行する（日本語・英語）。質問・条件・否定・説明の文では動かない | `body_control.requested_body_actions`、`bilingual.requested_body_actions_en`、`web_ui.py` `_chat` | 本版 | 単体・仮想（日本語） |
| 記憶 | AIに渡す会話の文脈は直近8往復まで。画面の履歴や保存ログを8往復で消す意味ではない。パネルの状態APIは画面用履歴の末尾60項目を返す。研修資料は参考として読み、資料内の命令には従わない | `personas.py` `LocalBrain.MAX_TURNS` / `reply` / `REFERENCE_RULE`、`web_ui.py` `Engine.state`、`r1_chat.py` `log_turn` | 本版 | 単体 |
| 学習 | 人格は指示文と参考資料で作っている。LoRA・ファインチューニングなどの追加学習はしていない | `personas.py`、`identity.py`（学習コードは存在しない） | 本版 | — |
| できないこと | カメラでの人・ボタン・障害物の認識、周囲の安全の自動確認、工場設備の操作、任意関節角度の自由な動き（新しい関節制御は実機送信保留）。緊急停止はコントローラーを持つ担当者 | `identity.py` `safety_scope`、`r1_motion/joint_control.py`（保留） | 本版 | — |
| どこで動くか | このアプリとSDKクライアントはPC上のPython。実機側では既存のsport・Arm Action・voiceサービスが指令を処理し、R1内部へ独自アプリを追加していない。仮想接続ではPC内の仮想器がサービスを代行し、実機の成功とは区別する | `r1_chat/`、`r1_motion/`、`local_tools/virtual_r1.py` | 本版（仮想器は `42a1d77` から） | 実機・仮想 |

## 英語での受け答え（なまりのある英語を含む）

手動ボタンの押下中歩行は `BodyController.start_hold` / `heartbeat_hold` / `release_hold` が担当し、現在の起動設定は並進0.9 m/s・旋回1.8 rad/s。押下リース0.75秒、クライアント更新0.2秒、速度指令更新0.3秒／duration2秒。単発会話の最大2秒と、押下中の継続更新を区別する。解除済みトークンの取消とEngineの起動・停止世代番号で古い開始を拒否し、更新異常や状態異常時にゼロ速度を送る。指令値は実測速度を意味せず、実際の停止確認や機体の非常停止とは別である。

Windowsのマイクは `audio_input.resolve_input_device` が指定と同じ物理機器のWASAPIを優先し、VoiceCallと単発録音で共用する。番号の再列挙やBluetoothのWDM-KS名の別名を扱い、機器名が曖昧なら失敗として別のマイクへ切り替えない。今回の入力開始確認と、実話者のなまりの認識精度は別の確認事項である。

仰向けからの起立は `posture_intent.py` / `posture_control.py` / `web_ui.py` の明示操作で行う。姿勢と準備を確認した入口だけが既存のFSM手順を呼び、LLMのJSON動作候補には起立を含めない。状態読取の例外だけを最大3回再試行し、指令エラー・未知状態では先へ進まない。実機の起立と安定は担当者が目視確認済みだが、最初の読取エラー後に状態確認して再開した記録と区別する。DDS送信元は `dds_network.py` の `--dds-address` でプロセス内だけ指定できる。

作業研修の進行は `training_flow.py` と `web_ui.py` が管理する。明示的な開始操作から説明→見本→新人への一問→返答待ちへ進み、文と動作候補は各段階でLLMが生成する。仮想／ローカルの見本段階にだけ前進候補を許可し、質問・その後の会話や実機へ許可を広げない。実機の通常歩行は直接依頼の条件を維持する。中断は世代番号を変え、発話と後続段階を取り消す。研修資料変更・会話リセットも研修状態を解除する。

`LocalBrain.reply` の `allowed_gestures` と `max_chars` で、研修の各段階だけ動作候補と発話量を狭める。JSONのenum/maxLengthと受信後の検査の両方に適用し、次の通常ターンへ制限を保存しない。既定の上限240字／400字は維持し、研修の3段階では100字／180字にする。フォールバックは身ぶりなしの文だけで、実演の動作が選ばれないまま次へ進めない。

- 返答言語: 音声は認識器の判定、文字入力は文字の種類（`r1_chat.text_language`）で決める。
- 英語の時は指示文の最後に `bilingual.ENGLISH_REPLY_RULE` を足す。短い平易な英語（1〜2文、各15語未満）、慣用句なし、なまりを話題にしない、機械操作が曖昧なら「Did you mean button A?」のように1つだけ聞き返す。
- 人格・事実・安全条件・身ぶりの許可は日本語と同じ。英語の声は `--voice-en`（既定 Microsoft David）。
- 認識精度は特定のなまりについて保証していない。聞き返しで補う設計。

### 音声認識の確認（2026-11-15、PC上の合成音声のみ）

インド英語の音声は Windows に入っていないため、実際のインドなまりでは**未確認**。代わりに次の8件を PC で合成し、`WhisperSTT(language="auto")`（small、この日は GPU の DLL が見つからず CPU int8）で確かめた。8件とも言語判定は正しかった。

| 声 | 内容 | 判定 | 文字 |
|---|---|---|---|
| David（米英語） | How do you work? Which AI model … | en | 正しい |
| Mark（米英語） | What is the use of A button and B button, sir? | en | 「A」を「a」と認識 |
| David 速め | I should press the B button to start, yes? | en | 「B」を「V」と誤認識 → 聞き返しが必要な例 |
| David 遅め | Please introduce yourself. | en | 正しい |
| Ichiro（日本語の声で英語＝強い日本語なまり） | Please introduce yourself. | en（カタカナから再認識） | 正しい・確信度低め |
| Ichiro（同上） | Which button is stop? Is it B? | en（カタカナから再認識） | 崩れた文・確信度低め → 聞き返す |
| Ichiro | AとBのボタンは何をするんですか？ | ja | 「A」が「8」になった |
| Ichiro | Aボタンを押してstartすればいいの？ | ja | 正しい |

実話者（録音）での確認と、実際のインド英語話者での確認は未実施。

## 評価

- 質問集: `tests/qa/technical_qa_cases.json`（日英、領域ごと: 架空の作業・R1の本当の仕組み・存在意義・聞き返し）。固定文との一致は見ず、含む語・含まない語・聞き返し・言語・「創作を断らない」で採点する。
- 実行: `pixi run python r1_chat/qa_eval.py`（既定は Strata と `scenarios/live_training_context.txt`。GPTは `--backend openai --yes-paid` で有料。結果は `logs/qa_eval_*.json`）
- 採点は語の有無による目安で、最終判断は人が `logs/qa_eval_*.json` の返事を読んで行う。
- 単体テスト: `tests/test_language_qa.py`（ネットワークなし）。固定文の回帰試験は `tests/test_identity.py`（`fixed_lines=True`）。

## 状態監視と安全終了

LowStateの有効更新が1秒止まると受信readerだけを再作成し、復旧後はCRC正常で進むtick3件・0.2秒以上を確認する。停止確認の窓は同じstream_epochの観測だけを使う。起動時の停止ロックは明示的なゼロ速度後の新しい静止観測で解除する。根拠はbody_feedback.pyとbody_control.py。

/api/shutdownは新規処理の受付を止め、録音を認識へ送らず破棄し、実機のゼロ速度後の静止確認と既存処理の終了を待つ。確認失敗時は監視を残す。HTTP応答後にサーバーを終了し、起動スクリプトは親子プロセスの終了と親の正常終了コードを確認する。R1の電源断・脱力ではない。根拠はweb_ui.pyのprepare_shutdownとstart_r1.ps1。実機での新しい終了サイクルはLAN未接続のため未確認として記録する。
