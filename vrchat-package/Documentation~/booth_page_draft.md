# AliceSDF Kit — BOOTH 商品ページ草案 (2026-09-19、価格 5,000 円 user 決定)

前身の AliceMochi 単品草案 (2,980 円案) は本 Kit に吸収 (Mochi 単品を別 SKU にするかは後日) 画像素材: `mochi_desktop.gif` / `wall_desktop.gif` / `terrain_desktop.gif` + 09-18 の 6 枚 + 統合 scene の実機スクショ (`vrchat_kit_demo_20260919.png`)

---

## 商品名
**AliceSDF Kit — 数式でできた、触れる VRChat ワールドギミック 7 種 (餅 / 凹む壁 / 掘れる地形 / 飾り 4)**

## キャッチ (1 行)
ポリゴンじゃない、数式でできた「触れるもの」 掴める餅、殴ると凹む壁、掘って盛れて立てる地形 置いてビルドするだけ、PC でも Quest でも

## 説明文 (日本語)

**置くだけ** prefab を scene に drag して root を置きたい場所へ → Build 他の package は不要 (VRChat Worlds SDK 3.7 以降、UdonSharp は SDK 同梱)

**入っているもの (prefab 9 個)**
| | 何 | 遊び方 (VR / デスクトップ) |
|---|---|---|
| Mochi (Mochi / Slime / Water の 3 preset) | 掴む・千切る・くっつく・体で押せる、やわらかい塊 | 掴む: 手を入れる / 左クリック押しっぱなし、分裂: グリップ / 右クリック、合体: 押し合わせる、体当たり: 体の形に凹む |
| Wall | 殴ると凹んで、ゆっくり元に戻る壁 | 殴る: 手 / 左クリック押しっぱなし、歩いて押す |
| Terrain | 盛って掘れて、その上に立てる地面 | 左手 / 左クリックで盛る、右手 / 右クリックで掘る、作った地形の上を歩く |
| Decor Basic / Cosmic / Fractal / Mix | 中に入れる飾り: 球と床 / 公転する惑星系 / メンガーのスポンジ / 混合 scene | 歩き回る、固い所からは押し出される |

- 全員同じものを見る (餅・凹み・地形は同期、owner 権威)
- **どこにでも置ける、回せる**: 法則は prefab に追従、描画は自分の cube の中だけなので複数置いても地面同士が喧嘩しない
- **調整は Inspector 1 箇所** (material を触らない): 色 / **自分の画像を貼る** (テクスチャを drop、UV なしで投影) / 大きさ / 粘り / 重さ / 凹み半径 / 回復速度 / ブラシ半径 / 容量 …
- **PC + Quest**: Quest では shader が自動で軽い予算に切り替わる (見た目の法則は同じ) Quest 向け world は画面内の volume を 1-2 個に (飾り 4 はモバイル GPU には重い)

**技術**: レイマーチング SDF (距離関数) で描画と当たり判定が同じ数式 = 見た目通りに触れる、Polygons: 0 オープンソースの ALICE-SDF (github.com/ext-sakamoro/ALICE-SDF) の 7 sample を、設定済み prefab + preset + 日英ドキュメント + サポート付きで製品化したものです

**動作環境**: PC (Windows) と Quest の VRChat ワールド (Quest 2 実機で確認済) SDK 3.7 以降推奨、Unity 2022.3

**実物を見る**: VRChat で「Mochi」(Community Labs、餅単体) / 「AliceSDF Kit Demo」(全部入り、PC は 9 個、Quest は飾り無し)

**同梱物**: `AliceSDFKit_x.y.z.unitypackage` (script 8 / shader 7 + cginc 2 / material 9 / prefab 9 / README 日英 / LICENSE)

**サポート**: BOOTH メッセージで受付 SDK バージョン・やったこと・期待した動き・`[Mochi]` / `[Wall]` / `[Terrain]` / `[SDF]` で始まる log 行 (Log Events を on) を添えてください 新規 Worlds project で再現できるものから優先して対応

**更新**: version は ALICE-SDF の vrchat-package と同期 (購入者は無償更新)

## 説明文 (English)

**Drop in and build.** Drag a prefab into your scene, place its root, Build. No other packages (VRChat Worlds SDK 3.7+, UdonSharp ships with the SDK).

**In the box (9 prefabs)**: Mochi (three presets: grab / split / merge / walk in), Wall (punch it, it dents and heals), Terrain (build and dig, and stand on what you made), four walk-in decor pieces (sphere on a plane, an orbiting planet system, a Menger sponge, a mixed scene). Synced for everyone.

**Place anywhere, turn as you like**: the law follows the prefab and each one draws only inside its own cube, so several share one world. **Tune from one Inspector**: colours, your own textures (drop an image in, projected without UVs), size, stickiness, weight, dent radius, recovery, brush radius, capacity.

**PC and Quest**: on Quest every shader switches to a lighter budget by itself (same look; tested on a Quest 2). Keep one or two volumes in view on Quest; the decor pieces are heavy on a mobile GPU.

**Tech**: raymarched SDF, one formula for rendering and collision, zero polygons. Productised from the seven open-source ALICE-SDF samples with configured prefabs, presets, EN / JP docs and support.

**Requirements**: PC and Quest VRChat worlds, SDK 3.7+, Unity 2022.3. Try it: search "Mochi" (Community Labs) or "AliceSDF Kit Demo".

## 画像 (順番案)
1. `mochi_desktop.gif` (11 s、掴む / 千切る / くっつく / 体当たり) — 1 枚目は動く物
2. `wall_desktop.gif` (壁殴り) 3. `terrain_desktop.gif` (穴掘り)
4. 統合 scene の実機スクショ `vrchat_kit_demo_20260919.png` (餅 + 壁、PC) — 「1 world に全部」
5. preset 3 種 `booth_presets_clientsim_20260918.png` 6. Inspector `booth_inspector_20260918.png`
7. 「Polygons: 0 | Resolution: INFINITE」 `booth_sign_polygons0_20260918.png` 8. checker テクスチャ `vrchat_product_texture_20260918.png`
9. Cosmic の惑星系 `kit_demo_cosmic_clientsim_20260919.png` (飾りの例)

## 価格 (user 決定 09-18)
- **5,000 円** (税込、単一 SKU、prefab 9 + 日英 README + サポート) BOOTH の表記は 5,000 か 4,980 か user 判断
- 無料 lite (Mochi 3 個固定 / 色固定 / texture 無し) を別商品で置いて導線にする案は継続
- Mochi 単品 (2,980 円案) を別 SKU にするかは後日

## タグ案
VRChat / ワールド / ギミック / Udon / UdonSharp / シェーダー / SDF / レイマーチング / 餅 / スライム / 壁 / 地形 / インタラクティブ / Quest対応
