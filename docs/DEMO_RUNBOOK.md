# R1 日本語チャット デモ運用手順書

対象：Unitree R1 ＋ WWG-R1（PC で音声認識 → OpenAI → 日本語音声を R1 のスピーカーで再生）
前提：プログラムはすべて PC で動かす。R1 の Onboard PC には触らない。R1 の体はこのデモでは動かさない（音声のみ）。

---

## 0. 役割分担（最低 2 人）

| 役割 | やること | 持ち物 |
|---|---|---|
| **オペレーター**（スタッフ） | R1 の起動・起立・停止、異常時の緊急停止 | コントローラー（E-Stop 付き） |
| **PC 担当** | LAN 設定、会話プログラムの起動、録音操作、ログ確認 | PC、LAN ケーブル、マイク |
| （いれば）**説明担当** | 来場者への声かけ、話し方の案内 | — |

> PC 担当とオペレーターは兼任しない。R1 に動作の命令を送るときは必ずオペレーターがそばにいる。

---

## 1. 事前準備（前日まで・PC 担当）

- [ ] `git clone https://github.com/suryarout16-ctrl/WWG-R1.git` → `cd WWG-R1` → `pixi install`
- [ ] `pixi run verify-sdk` → `unitree_sdk2py & cyclonedds loaded successfully!`
- [ ] API キーを保存：`pixi run python r1_chat/credentials.py set` → `... status` で「保存済み」
- [ ] （GPU を使う場合）環境変数 `R1_CUDA_DLL_DIR` に cuBLAS 12 / cuDNN 9 の DLL フォルダ
- [ ] R1 なしで通し確認：`pixi run chat --local`（PC のスピーカーで会話できる）
- [ ] 初回のみ音声認識モデル（約 480MB）が自動ダウンロードされる → ネットのある場所で 1 回起動しておく
- [ ] PC の充電、USB-LAN アダプタ（LAN ポートが無い PC）

---

## 2. 当日の立ち上げ（目安 10 分）

### 2-1. R1 の起動（オペレーター）

1. R1 を **仰向け** にするか **スタンド／ガントリーに吊る**。頭の周りを空ける（起動時に首が動く）
2. バッテリー残量 10% 以上を確認
3. バッテリーの電源ボタン「1 回押し → 2 回目長押し」
4. **「Zero Torque Mode. StartUp Completed.」** と話す／胸の LED が紫になるまで触らない
5. コントローラーも「1 回押し → 2 回目長押し」（ビープ 2 回）

### 2-2. LAN 接続（PC 担当）

1. LAN ケーブル：R1 の **うなじのポート** ↔ PC
2. PowerShell（リポジトリのルートで）：
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_lan.ps1
   ```
   名前 `Ethernet`・IP `192.168.123.164/24`・プライベート・ping・日本語の発話テストまで自動。最後に R1 が「接続できました」と話せば OK
3. 記録：`logs\setup_lan_<日時>.log`

### 2-3. 起立（オペレーター）

デモは **Running（白）** の状態で行う。コントローラーで：

| R1 の状態 | 操作 |
|---|---|
| 仰向け | L2+B（Damping・オレンジ）→ L2+↑（Locked Stand・青）→ **L2+X 長押し**（起き上がって Running・白） |
| 吊っている | L2+B → L2+↑ → 接地させ、1 人が肩を支える → **R2+A**（Running・白）※ガイド内に R1+A の表記もあるのでスタッフに確認 |

PC から「直立（Locked Stand）」までなら：`pixi run python r1_motion/stand.py -i Ethernet`（各段で確認入力あり。Running へはコントローラーで）
今のモードを読むだけ：`pixi run python r1_motion/stand.py -i Ethernet --status`

### 2-4. 会話プログラムの起動（PC 担当）

```powershell
pixi run chat -i Ethernet --volume 100 --gain 9
```

- 「[R1] 接続OK」→「[STT] 準備OK (cuda)」→「=== R1 チャット開始 ===」まで約 30 秒
- `(cpu)` と出たら GPU が使えていない（動くが認識が約 2 倍遅い）

### 2-5. 本番前の最終確認（1 分）

- [ ] 文字で「聞こえていますか」→ R1 が日本語で返事をする
- [ ] Enter → 「マイクテスト」と話す → Enter → 認識結果が正しく表示される
- [ ] 音量・聞こえ方を説明担当の位置で確認（小さければ `--gain 12`、割れるなら `--gain 6` で起動し直す）

---

## 3. デモ中の進行

### 来場者への案内（説明担当）

1. 「このロボットに日本語で話しかけてみてください。合図したら 1 文だけ、はっきりどうぞ」
2. マイクから 30〜50cm、ゆっくり 1 文（長い話・複数人同時は聞き取りにくい）
3. R1 の返事を待つ（話し終わりから 4〜6 秒）

### PC の操作（PC 担当）

| したいこと | 操作 |
|---|---|
| 声で話してもらう | 何も打たずに **Enter** → 来場者が話す → 話し終わったら **Enter** |
| 文字で代わりに入力 | 文字を打って **Enter**（騒がしいとき・聞き取りに失敗したとき） |
| 終了 | `q` → Enter |

- 1 往復の目安：認識 約 1 秒 ＋ 返事の生成 約 3〜4 秒 ＋ 発話
- 「（聞き取れませんでした）」「短すぎるので無視」→ もう一度 Enter から
- 通信が切れても止まらない（R1 が「通信がうまくいきません」と話す）

### 話題の例（反応が良いもの）

- 「君は誰？」「何ができるの？」「好きな食べ物は？」「今日の調子は？」
- 体を動かす依頼（「踊って」「歩いて」）には「今は話すことだけ」と答える設定

---

## 4. トラブル対応

| 症状 | まずやること |
|---|---|
| **R1 の様子がおかしい・止まらない・転びそう** | **オペレーターが即 E-Stop**（無ければ L2+B を 5 秒以上長押し）。PC で止めようとしない |
| ping が通らない | R1 の起動完了（紫・音声）を確認 → ケーブル差し直し → `setup_lan.ps1` を再実行 → `logs\setup_lan_*.log` を確認 |
| 「R1 の音声サービスに届きません」「code=3102」 | `-i Ethernet` の名前、ネットワークが「プライベート」か、R1 の起動完了を確認 |
| R1 から音が出ない（PC 側は「発話OK」） | R1 の音量（アプリ）、別のプログラムが再生中でないか。`--say "テスト"` で単体確認 |
| 声が小さい | `--volume 100` は上限。`--gain 12` で起動し直す |
| 認識がおかしい | マイクに近づく・1 文ずつ。`--save-audio` で録音を残して確認 |
| 返事が遅い（10 秒以上） | 会場の回線を確認。続くなら `--model` を軽いモデルに |
| 「[LLM エラー]」 | 回線・API キー（`credentials.py status`）を確認 |
| 起動時「API キーがありません」 | `pixi run python r1_chat/credentials.py set` |
| `(cpu)` で起動した | `R1_CUDA_DLL_DIR` を設定して起動し直す（そのままでも動作は可） |

困ったときは **ログを見せる**：`logs\` の最新の `r1_chat_*.log` / `*_turns.jsonl` / `setup_lan_*.log` / `r1_motion_*.log`

---

## 5. 終了手順

1. PC：会話プログラムで `q`
2. オペレーター：R1 を寝かせる（L2+X 長押し → 自動で Damping）か、Locked Stand（L2+↑）にして吊る → L2+B
3. オペレーター：**L2+Y で Zero Torque（紫）** を確認してから電源オフ（バッテリーボタン 1 回押し → 長押し）
   - 立ったまま電源を切らない（全関節が脱力して倒れる）
4. コントローラーの電源オフ（ビープ 3 回）
5. PC：必要なら LAN を元に戻す `powershell -ExecutionPolicy Bypass -File scripts\setup_lan.ps1 -Restore`
6. PC：`logs\` を保存（会話内容を含むので共有範囲に注意）

---

## 6. 安全ルール（運用ガイドより・必ず守る）

- Onboard PC にログインしない。プログラムは PC から
- 初めて動かすプログラムは吊った状態で。床では周囲 2m 以上空ける
- ダンス／カンフー系のコマンド（R1+SELECT、R1+十字キー、R2+←）は入力しない
- バッテリー残量低下（黄緑の点滅）→ すぐ作業をやめて停止・充電
- 日本語の音声を流すこと、動作の命令を送ることはスタッフに確認してから

---

## 7. コマンド早見表（リポジトリのルートで）

| 用途 | コマンド |
|---|---|
| LAN 設定〜発話テスト | `powershell -ExecutionPolicy Bypass -File scripts\setup_lan.ps1` |
| LAN を元に戻す | `powershell -ExecutionPolicy Bypass -File scripts\setup_lan.ps1 -Restore` |
| R1 のモードを読む | `pixi run python r1_motion/stand.py -i Ethernet --status` |
| 一言話させる | `pixi run chat -i Ethernet --volume 100 --gain 9 --say "こんにちは"` |
| 聞き取り確認（おうむ返し） | `pixi run chat -i Ethernet --volume 100 --gain 9 --echo` |
| **本番** | `pixi run chat -i Ethernet --volume 100 --gain 9` |
| R1 なしで練習 | `pixi run chat --local` |
| API キーの確認 | `pixi run python r1_chat/credentials.py status` |
