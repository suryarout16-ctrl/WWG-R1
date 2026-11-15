# Unitree R1 開発メモ（2026-11-14 時点）

これは初期調査時の記録で、見込みや当時未検証だった機能を含む。現在の接続・動作確認・制約・起動終了手順は [ナレッジ集](KNOWLEDGE_BASE.md) と [即興デモ](LIVE_DEMO.md) を優先する。

出典: ハッカソン配布資料・運用ガイド / humanoid_hackathon_env (commit 5ae30a8, 11/13 20:03) /
Unitree公式 R1 video_service / syswonder/robot-unitree-r1 (commit 8e77682)

## 0. 絶対ルール
- Onboard PC に SSH しない・中で何も動かさない。プログラムは全部自分のPCから LAN 経由
- 初めて動かすコードはスタンド/ガントリーで吊って試す。床で動かすなら周囲 2m 以上空ける
- 動作コマンドを送るときは、コントローラーを持ったオペレーターを別に1人つける
- 緊急停止: コントローラー底面の E-Stop（ないとき L2+B を5秒以上長押し）
- ダンス/カンフー系コマンド（R1+SELECT, R1+十字, R2+←）は禁止
- G1 用 LocoClient の FSM 系関数（Lie2StandUp など）を R1 に使わない（番号の意味が違う）

## 1. 接続手順
1. LAN ケーブル: R1 のうなじのポート ↔ PC の有線LAN
2. 有線LAN に固定IP: `192.168.123.164` / `255.255.255.0` / ゲートウェイ空欄（R1 は `192.168.123.161`）
3. `ping 192.168.123.161` → 損失0%
4. `cd ~\humanoid_hackathon_env` → `pixi run sample-tts -i Ethernet -s 1 -v 70 -t "Connection test"`
   → `[Info] TTS command succeeded.` で接続完了
5. 3102 エラー: `-i` 名の間違い / ネットワークが「パブリック」(→プライベートに) / R1 起動未完了

## 2. できること / できなさそうなこと
### すぐできる（配布環境・PC側だけで完結）
- 発話 TTS（英語 speaker 1 / 中国語 0。日本語不可）、音量取得/設定 — G1 の AudioClient を流用
- 歩行の速度指令 `Move / SetVelocity / StopMove`（API 7105。G1/R1 で同じ API なので sample_walk は届く見込み。ただし要スタッフ相談）
- 状態の読み取り（DDS `rt/lowstate`: IMU・関節角・バッテリー） — 読むだけなので安全

### 工夫すればできる（要スタッフ相談）
- R1 専用 LocoClient（Start/Damp/WaveHand/ShakeHand など）: パッケージに入っていないが、
  `r1_loco_client.py` を手元にコピーして `from unitree_sdk2py.rpc.client import Client` に直せば使える
- 両腕・腰・頭の関節制御 `rt/arm_sdk`（FSM 811 中、13関節、100Hz 送信 + CRC）。腕はゆっくり（0.05 rad/s 程度）
- カメラ・深度: Stereo patch PC1 を ON + video_hub を OFF にしたうえで UDP 5001(1280x720 H.264) /
  5002・5003(左右 544x448) / 深度 /dev/video-dep(544x448)。UDP の宛先や深度クライアントは Onboard 側の設定が絡むので運営次第
- マイク: マルチキャスト 239.168.123.161:5555（16kHz mono）。アプリで音声ウェイク+マイク有効が必要

### できない / やめておくべき
- 脚の関節を直接動かす・姿勢制御を自作する（Debug Mode → 高レベルサービス停止、転倒リスク大）
- SLAM / 地図（R1 EDU + Mid360 LiDAR + unitree_slam が必要。未搭載）
- 正確な距離・角度移動（開ループ。0.3 m/s 前進 / -0.4 m/s 後退のみ実機確認例あり、横移動・旋回は未検証）
- 把持・指先位置精度（手はなし、手首 roll まで。IK の精度も未検証）
- カメラのキャリブレーション済み 3D（公式 cam_param.txt が 9080 番で取れれば可）
- R1 本体の TTS で日本語を話す（PC で合成して PlayStream で流せば可。このリポジトリの方式）
