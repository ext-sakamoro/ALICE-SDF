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

## 画像 (順番案)
1. `mochi_desktop.gif` (11 s、掴む / 千切る / くっつく / 体当たり) — 1 枚目は動く物
2. Labs world のスクショ (餅 5 個 + 看板)
3. preset 3 種を並べた 1 枚 (今日の ProductTest scene: 白 / 緑 / 青、Build & Test 中に撮る)
4. Inspector のスクショ (Look / Mochis の knob が見える)
5. 「Polygons: 0 | Resolution: INFINITE」の看板アップ
6. checker テクスチャを貼った Slime (今日の `vrchat_product_texture_20260918.png`) — 「自分の画像が貼れる」の証拠

## 価格 (案、user 判断)
- **AliceMochi**: 1,500 円 (VRChat ワールドギミックの相場 1,000〜3,000 円の中位、prefab 3 + サポート)
- **無料 lite** (餅 3 個固定・色固定・サポート無し) を別商品で置くと導線になる — 2 段にするか単一有料かは user
- 上位 SKU (TerrainSculpt / DeformableWall を足した「SDF Interactive Set」) は後日、3,000〜4,000 円帯

## タグ案
VRChat / ワールド / ギミック / Udon / UdonSharp / シェーダー / SDF / レイマーチング / 餅 / スライム / インタラクティブ / PC only
