# AliceMochi — BOOTH 商品ページ草案 (2026-09-18)

価格 / 販売形態は user 判断 (案は末尾) 画像素材: `mochi_desktop.gif` / `wall_desktop.gif` / `terrain_desktop.gif` (後 2 本は上位 SKU 用) + Labs world `Mochi` のスクショ

---

## 商品名
**AliceMochi — 掴める・千切れる・くっつく、やわらかい餅 (VRChat ワールド用)**

## キャッチ (1 行)
ポリゴンじゃない、数式でできた餅 掴んで、千切って、くっつけて、体でぐにゃっと押せる VR でもデスクトップでも

## 説明文 (日本語)

**置くだけ** prefab を scene に drag → Build & Test で動きます 他の package は不要 (VRChat Worlds SDK 3.7 以降、UdonSharp は SDK 同梱)

**遊べること**
- 掴む: VR は手を餅に入れる、デスクトップは左クリック押しっぱなし
- 千切る: VR はグリップ、デスクトップは右クリック (体積は保存、半分ずつになる)
- くっつける: 餅同士を押し合わせると合体して大きくなる
- 体当たり: 歩いて突っ込むと体の形に凹んで、餅が逃げる
- 全員同じ餅を見る (同期)、最大 16 個

**調整は Inspector 1 箇所** (material を触る必要なし)
色 (餅 / ハイライト / 地面) / **好きな画像を貼る** (餅・地面それぞれにテクスチャを drop、餅の模様は動いても付いてきて、千切ると小さく、合体で大きくなる) / 最初の個数・大きさ・並び / 粘り (Blend K) / 重さ / 分裂の可否 / VR の掴みルール

**preset 3 種** — Mochi (白い餅) / Slime (緑、よく伸びる) / Water (青、さらさら) を prefab で同梱、色と粘りを変えれば何にでもなります

**技術**: レイマーチング SDF (距離関数) で描画と当たり判定が同じ数式 = 見た目通りに触れる、Polygons: 0 オープンソースの ALICE-SDF (github.com/ext-sakamoro/ALICE-SDF) の Mochi sample を、設定済み prefab + preset + 日英ドキュメント + サポート付きで製品化したものです

**動作環境**: PC (Windows) の VRChat ワールド Quest / Android 非対応 (レイマーチングシェーダー) SDK 3.7 以降推奨、Unity 2022.3

**実物を見る**: VRChat で「Mochi」を検索 (Community Labs) → `https://vrchat.com/home/world/wrld_0cb72970-948e-4212-b955-fd3dd567aa42`

**同梱物**: `AliceMochi_x.y.z.unitypackage` (script 1 / shader 1 / material 3 / prefab 3 / README 日英 / LICENSE)

**サポート**: BOOTH メッセージで受付 SDK バージョン・やったこと・期待した動き・`[Mochi]` で始まる log 行 (Log Events を on) を添えてください 新規 Worlds project で再現できるものから優先して対応

**更新**: version は ALICE-SDF の vrchat-package と同期 (購入者は無償更新)

## 説明文 (English)

**Drop in and build.** Drag the prefab into your scene, Build & Test. No other packages (VRChat Worlds SDK 3.7+, UdonSharp ships with the SDK).

**Play**: grab (hand in VR, hold left click on desktop), split (grip / right click, volume conserved), merge (push two together), walk in (your body dents it and it slides away). Synced for everyone, up to 16 mochis.

**Tune from one Inspector**: colours, **your own textures** on the mochis and the ground (drop an image in; the mochi pattern follows, shrinks and grows with each mochi), starting count / size / layout, stickiness, weight, whether they split, VR grab rules. Three presets: Mochi, Slime, Water.

**Tech**: raymarched SDF, one formula for rendering and collision, zero polygons. Productised from the open-source ALICE-SDF Mochi sample with configured prefabs, presets, EN / JP docs and support.

**Requirements**: PC VRChat worlds only (no Quest / Android), SDK 3.7+, Unity 2022.3. Try it: search "Mochi" in Community Labs.

## 画像 (順番案、6 枚撮影済 09-18、実体は Unity project `Mochi/Assets/Screenshots/`)
1. `mochi_desktop.gif` (11 s、掴む / 千切る / くっつく / 体当たり) — 1 枚目は動く物
2. Labs world のスクショ `vrchat_mochi_published_20260917.png` (餅 + 実 client)
3. preset 3 種を並べた 1 枚 `booth_presets_clientsim_20260918.png` (左から Water / Slime / Mochi、ClientSim 1920x1080、各 prefab の cube 底面が地面パッチ)
4. Inspector のスクショ `booth_inspector_20260918.png` (Look / Placement / Mochis / Mochi Settings / Interaction / Desktop / Debug の全 knob、Slime に checker texture 割当状態、560x1000)
5. 「Polygons: 0 | Resolution: INFINITE」HUD 入りの餅 3 個 (合体中) `booth_sign_polygons0_20260918.png` (sample scene の HUD 帯、2042x1310) — 看板は screen-space HUD で world-space 看板ではない (Labs world の 3D sign は MOCHI / 操作説明のみ)
6. checker テクスチャを貼った Slime `vrchat_product_texture_20260918.png` (実 client) — 「自分の画像が貼れる」の証拠

## 価格 (案、user 判断 — 09-18 user「1,500 円は安くない?」に対する私の再提案)
- 初案 1,500 円は BOOTH ワールドギミック相場 (1,000〜3,000 円) の中位に置いただけで、本品の固有性を織り込んでいない
- **推奨 2,980 円** (税込、単一有料 SKU) 根拠: (1) 同種品が BOOTH に無い (ポリゴン餅 / スライム系 asset は mesh + Rigidbody / cloth で「千切れる・合体」が無い) (2) prefab 3 + 日英 README + サポート付き (3) PC only は Quest 層を落とすので 4,000 円台は売れ行きが鈍る (4) 2,980 は「ギミック 1 個」の心理上限 3,000 円の直下
- 無料 lite (餅 3 個固定 / 色固定 / texture 無し / サポート無し) を別商品で置いて導線にする — 有料版との差分を Inspector で見せられる
- 上位 SKU (TerrainSculpt / DeformableWall を足した「SDF Interactive Set」) は後日 4,980 円帯、単品購入者には差額 upgrade を BOOTH メッセージで案内
- 初期 2 週間は 2,480 円の発売記念価格で reviews を集めてから 2,980 に戻す運用も可 (BOOTH は価格変更自由)

## タグ案
VRChat / ワールド / ギミック / Udon / UdonSharp / シェーダー / SDF / レイマーチング / 餅 / スライム / インタラクティブ / PC only
