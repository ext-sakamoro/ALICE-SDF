# ALICE-SDF for VRChat

**「ポリゴンよ、さらば。」**

ALICE-SDFは、**数式（SDF）で定義された無限に滑らかな曲面** をVRChatの世界に持ち込むためのオールインワン・パッケージです。
単に見えるだけでなく、**プレイヤーがその上に立ち、衝突し、インタラクトする** ことが可能です。

[English / 英語版](README.md)

コーディングエージェント (Claude Code / Codex / Cursor) に導入させる場合は [`AGENTS.md`](AGENTS.md) (英語) を読ませてください 手順ごとの確認方法、Unity なしで検証できる範囲、Mochi をアバター / 小道具に転用する時の「法則 / 結線」の分離、既知エラーと対処が書いてあります

---

## なぜ ALICE-SDF なのか？

従来の「3Dメッシュ」によるワールド制作には限界がありました。ALICE-SDFは、**数式** を送ることで世界を描画します。

1. **無限の解像度**: どれだけカメラを近づけても、ポリゴンの角（カクつき）は一切見えません。完全な球体、滑らかな結合が可能です。
2. **圧倒的なデータ圧縮**: 複雑な有機的形状も、数式ならわずか数キロバイト。ワールド容量を劇的に削減します。
3. **リアルタイム変形**: シェーダーパラメータを変えるだけで、形状がグニャグニャとモーフィングします。メッシュの頂点移動では不可能な表現です。
4. **物理判定 (Collider) 完全対応**: 「レイマーチングは当たり判定がない」という常識を覆しました。UdonSharpにより、数式の床の上に立つことができます。

---

## インストール

### Unity Package Manager 経由 (推奨)

1. Unityで **Window > Package Manager** を開く
2. **+** ボタン > **Add package from disk...** をクリック
3. ALICE-SDFフォルダ内の `package.json` を選択

または `Packages/manifest.json` に git URL を直接追加:

```json
{
  "dependencies": {
    "com.alice.sdf": "https://github.com/ext-sakamoro/ALICE-SDF.git?path=vrchat-package"
  }
}
```

### 動作環境

- Unity 2022.3.x (VRChat 推奨バージョン)
- VRChat Creator Companion で作った **Worlds** プロジェクト (SDK3 + UdonSharp、物理判定に必須 シェーダーだけなら無くても動く)
- プロジェクトのパスは ASCII のみ (日本語などを含むパスでは VRChat SDK の `UnityEventFilter` が Play 時に落ちる、トラブルシューティング参照)

---

## ALICE-Baker v0.3 (Deep Fried Edition)

**「数式を書くのが難しい？」心配無用です。**

付属の **ALICE-Baker** ツールが、JSON定義ファイルから「最適化されたシェーダー」と「Udonスクリプト」を全自動生成します。

1. Unityメニューから **Window > ALICE-SDF Baker** を開く
2. `.asdf.json` をドラッグ＆ドロップ（またはJSONテキストをペースト）
3. **Bake!** ボタンをクリック

これだけで、**Prefab** が生成されます。あとはシーンに置くだけです。

### "Deep Fried" 最適化とは？

VRChatのUdon VMは、通常のC#に比べて実行速度に制約があります。ALICE-Baker v0.3は、生成されるコードに対して**狂気的な最適化（Deep Frying）**を施します。

| 最適化手法 | 効果 (対 通常実装比) |
|------------|----------------------|
| **命令融合 (Instruction Fusion)** | 計算式をインライン展開し、一時変数のメモリ確保を削減。 |
| **除算の悪魔払い (Division Exorcism)** | 重い「割り算」命令を排除。逆数（`1/k`）を事前計算し「掛け算」に変換。 |
| **Udonスカラ展開** | `Vector3` の構造体コピーはUdonでは高コストです。これを `float x,y,z` のスカラ演算に分解し、VMオーバーヘッドを回避。 |
| **関数コール削除** | `Sdf.Union()` などの関数呼び出しを `Mathf.Min()` へ直書き展開。メソッド呼び出しコストをゼロに。 |
| **Smooth演算の完全展開** | `SmoothUnion/Intersection/Subtraction` を数式レベルでインライン化。`inv_k` をコンパイル時定数として埋め込み。 |

これにより、**50HzのFixedUpdate内でも余裕で動作する** 高速な物理判定を実現しました。

---

## 3つのコアコンポーネント

### 1. ALICE-Shader (描画エンジン)

ピクセル単位で距離関数をレイマーチング描画します。

- **SV_Depth 対応**: VRMアバターや他のオブジェクトとの前後関係（オクルージョン）も完璧に処理します。「アバターが壁に埋まると描画がおかしくなる」というレイマーチング特有の問題を解消。
- **Deep Fried LOD**: カメラ距離に応じて、レイのステップ数と精度を動的に調整します。遠くのオブジェクトは負荷が激減します。

| 距離 | ステップ | Epsilon | 品質 |
|------|---------|---------|------|
| < 20m | 128 | 0.0001 | High |
| 20–60m | 64 | 0.001 | Medium |
| > 60m | 32 | 0.005 | Low |

### 2. ALICE-Udon (物理エンジン)

シェーダーと同じ数式をC#で計算し、プレイヤーを押し出します。

- プレイヤーが数式の内部（`d < 0`）に侵入すると、法線ベクトル（`∇SDF`）を計算し、表面まで押し戻します。
- 1回のティックにつき SDF評価 1回 + 勾配サンプル 6回 = **計7回の評価のみ**。典型的なコストは **< 0.01ms/プレイヤー**。
- ネイティブプラグイン（DllImport）を使用しないため、**Quest単体でも動作可能なロジック** です。

### 3. ALICE-Math (共通ライブラリ)

HLSLとUdonSharpの両方で「全く同じ挙動」をするように設計された数学ライブラリ群。

- `SmoothUnion` や `Twist` などの形状操作も、見た目と当たり判定が1ミリもズレません。
- 15プリミティブ + 17演算の全てが HLSL / C# で1:1対応。

---

## サンプル (SDF Gallery)

7種類のサンプルワールドを同梱しています。**Package Manager > Samples** タブからインポートしてください。

| サンプル | 概要 | SDF式 |
|---------|------|-------|
| **Basic** | 地面 + 浮遊する球体。最もシンプルなSDFワールド。 | `min(plane, sphere)` |
| **Cosmic** | アニメーション付き太陽系 — 太陽、軌道惑星、傾斜リング、月、小惑星帯。 | `SmoothUnion(sun, planet, ring, moon, asteroids)` |
| **Fractal** | メンガーのスポンジ迷宮の内部を歩けます。ねじり変形付き。 | `Subtract(Box, Repeat(Cross))` — 1つの式で無限の複雑さ |
| **Mix** | Cosmic × Fractal 融合 — フラクタル惑星 + トーラスリング + 玉ねぎシェル。 | `SmoothUnion(Intersect(Sphere, Menger), Torus, Onion(Sphere))` |
| **DeformableWall** | 触る・マウスで殴る・突っ込むと壁が凹む。凹みは時間で回復。 | `min(ground, SmoothSubtract(wall, dent_spheres...))` |
| **Mochi** | ぷにぷに餅ブロブ。掴む・合体・分裂・巨大化。SmoothUnion軟体物理。 | `SmoothUnion(ground, SmoothUnion(mochi1, mochi2, ..., k))` |
| **TerrainSculpt** | VRの手やマウスで地形を掘る・盛る。掘った穴に本当に落ちる。**SDFでしか不可能。** | `SmoothUnion(SmoothSub(plane, digs...), hills...)` |

各サンプルには以下が含まれます：
- `*_Raymarcher.shader` — SV_Depth / LOD / AO / フォグ対応レイマーチングシェーダー
- `*_Collider.cs` — UdonSharpコライダー（`#if UDONSHARP` ガード付き）
- `*.asdf.json` — Baker用の定義ファイル

### インタラクティブサンプル (VR)

**DeformableWall**、**Mochi**、**TerrainSculpt** は、VRハンドトラッキングによるリアルタイムSDF変形を実演するサンプルです。上記の静的サンプルとは異なり、毎フレーム UdonSharp から `Material.SetVectorArray` でシェーダーに動的データを送信します。

#### DeformableWall — 触って凹む壁

地面の上に立つ平面の壁。VRの手で触る、マウスで殴る、歩いて突っ込む — 接触点に凹みが発生し、時間の経過とともに徐々に回復します。凹みは本物の形状で、当たり判定も凹んだ壁に対して行われます。

**仕組み:**
1. 手（デスクトップは左ボタン押下中の視線カーソル）が **凹む前の壁面** から `Impact Distance` 以内で、かつ生きている凹みの空洞の中でない時に衝撃を登録 — 出来たての凹みに手を追従させても 0.4 m の壁を貫通しない 生きている凹みの中心から半径の半分以内を叩くと slot を消費せずその凹みを回復前に戻す
2. 各凹みは (位置, 強度)。強度は 1 から `exp(-Decay Speed * t)` で回復、0.01 未満で slot 解放（同時に最大 16、満杯なら最も弱いものを置換）
3. 毎フレーム、配列を壁の寸法 / `Dent Radius` / `Dent Smoothness` と共に `Material.SetVectorArray("_ImpactPoints", ...)` でシェーダーに送信 — 当たり判定と描画がずれない
4. シェーダーが `opSmoothSubtraction(wall, sphere(Dent Radius * strength))` で各凹みを刻み、同じ方法で自分の体の capsule も壁に押し込む
5. collider はプレイヤーの体を足元から目まで sample し、最も深い点を横方向に押し出す（上には持ち上げない、dead band で押しが止まる）

**操作:**

| 操作 | VR | デスクトップ | 結果 |
|------|----|------------|------|
| **凹ませる** | 手を壁面に近づける | 左クリックを押したまま壁を見る | 接触点に凹み、新しいうちは光る |
| **叩き続ける** | 同じ場所を叩き続ける | ボタンを押したまま | 凹みが最大深さに戻る（slot 1 個、半径以上は深くならない） |
| **たくさん** | あちこち叩く | ボタンを押したまま視線を動かす | 同時に最大 16、最も弱いものから再利用 |
| **回復** | 待つ | 待つ | 凹みが平面に戻る |
| **もたれる** | 壁に歩いて突っ込む | 壁に歩いて突っ込む | 体の capsule 形の溝が壁に入り、押し戻される |

**Inspectorパラメータ:**

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| Wall Width / Height / Thickness | 5 / 2.5 / 0.2 | 壁の半サイズ（毎フレーム shader に送る） |
| Impact Distance | 0.08 | 凹みが発生する手 / カーソルと（凹む前の）壁面の距離 |
| Impact Cooldown | 0.15秒 | 同じ手からの連続衝撃の最小間隔 |
| Decay Speed | 0.5 | 回復: 強度が exp(-speed * t) で減衰 |
| Dent Radius | 0.35 | 強度 1 の時の凹みの半径 |
| Dent Smooth | 0.08 | 凹みの SmoothSubtraction ブレンド係数、毎フレーム material に送る |
| Cursor Max Dist | 4.0 | デスクトップ: 視線で壁を探す距離 |
| Collision Margin / Push Strength / Body Samples | 0.1 / 1.0 / 5 | プレイヤー押出: 体を足元から目まで sample |
| Player Radius / Body Dent K | 0.3 / 0.12 | shader が壁に押し込む体の capsule（自分だけに見える） |
| Log Events | off | 衝撃 / クリック miss / 押出 / owner 取得 / 受信ごとに `[Wall] ...` を `Debug.Log` 1 行 — VRChat client の `output_log_*.txt` を grep |

**シェーダーパラメータ:** `Light Direction`、`Enable Soft Shadow`（壁が地面に落とす影）、`Fog Density` レイマーチャは Mochi と同じ最接近点採用と hard union AO（壁の縁に暗い筋が出ず、凹みの周りに暗いリングが出ない）

#### Mochi — 掴む・合体・分裂・巨大化

![デスクトップでの Mochi: 歩いて入ると体の形に凹み、クリックで掴み、右クリックで分裂、運んで合体](Documentation~/mochi_desktop.gif)

**VRChat で遊ぶ:** このサンプルは world **Mochi** として公開しています — [vrchat.com/home/world/wrld_0cb72970-948e-4212-b955-fd3dd567aa42](https://vrchat.com/home/world/wrld_0cb72970-948e-4212-b955-fd3dd567aa42) private (テスト中) の間は作者と招待したフレンドだけ入れます: web の world ページで **Launch** (インスタンスを作って VRChat が開く)、または VRChat のメニュー → **Worlds** → 作者の world 一覧 → **Mochi** → **Launch** PC では Steam 経由で `vrchat://launch?ref=vrchat.com&id=wrld_0cb72970-948e-4212-b955-fd3dd567aa42:<任意の数字>~private(<自分の user id>)~region(jp)` でも同じ Community Labs / public になれば名前で検索できます PC 専用 (レイマーチャーは Quest 向けにビルドしていません)

地面の上に置かれた柔らかい餅（もち）ブロブ。掴んで引っ張って分裂させたり、押し合わせて合体させたり、どんどん大きくしたりできます — VR では手で、デスクトップではマウスで（上の GIF は VRChat client の 11 秒: 歩いて入ると体の形に凹み、クリックで掴み、素早く振ると分裂、別の餅に運ぶと合体）。

**仕組み:**
1. 最大16個の餅球を `(位置, 半径)` のペアで管理
2. すべての餅を `opSmoothUnion` でブレンド — 近くの餅同士は自然に一体化して見える
3. 地面との接触には別の `opSmoothUnion`（低い `k` 値）を使用 — 「床にぷにっと座っている」感触
4. UdonSharp が毎フレーム `Material.SetVectorArray("_MochiData", ...)` でシェーダーに送信

**VR操作:**

| 操作 | やり方 | 結果 |
|------|--------|------|
| **掴む** | 手を餅の中に0.08秒置く | 餅が手にくっつく |
| **移動** | 掴んだまま手を動かす | 餅が手に追従する |
| **分裂** | 持ったままグリップ（VR）/ 右クリック（デスクトップ） | 餅が2つに分裂（体積保存: `r_new = r * cbrt(0.5)`）、もう半分は掴んだ場所に残る |
| **リリース** | VR: 半径の4倍運んだところで手を素早く抜く（ゆっくりならすぐ掴み直す）/ デスクトップ: ボタンを離す | 餅が落下して地面に着地 |
| **合体** | 自由な餅同士を近づける | 1つの大きな餅に合体（`r = cbrt(r1^3 + r2^3)`） |
| **巨大化** | 合体を繰り返す | 餅がどんどん大きくなる |
| **歩いて押す** | 餅に歩いて入る | 体の形に凹み、質量比で餅が退き、自分は押し返される |

**デスクトップ:** 餅の上で Use（左クリック）を押し続けると、視線上で餅の中心に最も近い点が仮想の右手になり、上記の掴む / 移動 / 分裂 / リリース / 合体が視点移動で動きます。ボタンを離すと落とします。右クリック（desktop の InputDrop、VR はグリップ = InputGrab）で持っている餅を割ります。Log Events を on にすると、離した時に「持ったまま割ったか / どこまで運んだか」が 1 行出ます。

**Inspectorパラメータ:**

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| Blend K | 0.5 | 餅同士のSmoothUnion係数（大きいほど粘着）— 毎フレーム Material に送るので当たり判定と描画が常に一致 |
| Ground K | 0.15 | 地面とのSmoothUnion係数（ぷにぷに接地感）— 毎フレーム Material に送る |
| Min Radius | 0.1 | 最小餅サイズ（これ以下には分裂しない） |
| Grab Threshold | 0.8 | 掴むために手が半径の何割以内に入る必要があるか |
| Grab Dwell Time | 0.08秒 | 掴み発動までの滞在時間（誤掴み防止） |
| Split On Pull | off | 旧来の「引っ張ると千切れる」: 掴んだ位置から半径の2.5倍動くと分裂（デスクトップの cursor は視線を動かすだけで超えるため off） |
| Split Distance | 2.5 | Split On Pull 用の引っ張り距離（半径の倍率） |
| Release Distance | 4.0 | VR の手が餅を落とす運搬距離（半径の倍率）、デスクトップ cursor はボタンを離した時のみ |
| Merge Threshold | 0.7 | 自動合体が発生する距離（合計半径の割合） |
| Log Events | off | 掴む / 分裂 / リリース / 合体 / クリック / 押し の event ごとに `[Mochi] ...` を1行 `Debug.Log` — VRChat client の `output_log_*.txt` を grep すれば何が起きたか分かる |

**Materialパラメータ**（シェーダーのみ）:

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| Light Direction | (1, 1, -0.5) | wrap / diffuse 陰影の平行光源方向 |
| Enable Soft Shadow | 1 | 餅が地面に落とす接地影（LOD tier 別 32 / 16 / 8 step） |
| Shadow Softness | 16 | 半影の幅（大きいほど鋭い） |
| Shadow Max Distance | 10 | 影レイの長さ |
| Fog Density | 0.005 | 指数距離フォグ |

#### TerrainSculpt — 掘れる・積める地形

**VRChat史上初、掘った穴に本当に落ちる体験。**

Y=0の平面地形をリアルタイムにスカルプトできます — VRの手でも、マウスでも。描画もコリジョンも全く同じSDF数式で評価されるため、**掘った穴に実際に落ち、積んだ丘に実際に登れます**。

従来のVRChatでは、MeshColliderはランタイムに再計算できないため、これは原理的に不可能でした。ALICE-SDFは描画と物理の両方で同じ数式を評価するため、見た目=当たり判定が常に成立します。

**仕組み:**
1. ベース地形はY=0の地面（平面）
2. 盛る → `opSmoothUnion(terrain, sphere)` — 手 / カーソルの位置に丘を追加
3. 掘る → `opSmoothSubtraction(terrain, sphere)` — 手 / カーソルの位置に穴を掘削
4. 操作は循環バッファに記録（`Sculpt Capacity`、既定 96、最大 128）。満杯になると最古の操作を上書き
5. UdonSharp が毎フレーム操作配列をシェーダーに送信
6. 立つ: VRChat のプレイヤーコントローラは足元に Unity のコライダーが無いと接地できないため、スクリプトが小さな見えない箱 (`TerrainSupport`) を毎フレーム、足の幅 (`Foot Radius` の 5 点) の中で最も高い SDF 表面に水平に置きます (尾根は足が完全に外れるまで乗っていられる) 足元を掘れば箱ごと地形と一緒に下がって落ち、足元に盛れば新しい頂上に持ち上げられます 高い丘の急な側面は壁として押し返し、0.3 m 以下の段差はそのまま歩いて登れます

**操作:**

| 操作 | VR | デスクトップ | 結果 |
|------|----|------------|------|
| **掘る** | 右手を地面に近づける | 右クリックを押したまま地面を見る | 半球状の穴が掘れる。落ちる |
| **盛る** | 左手を地面に近づける | 左クリックを押したまま地面を見る | 丘/盛り土が出現。登れる |
| **深く掘る** | 右手を穴の中に保持 | 穴をもう一度右クリック | 操作ごとにさらに深く掘削 |
| **高く積む** | 左手を丘の上に保持 | 丘をもう一度左クリック | さらに地形を積み上げ |
| **なぞる** | 手を地面に沿って動かす | ボタンを押したまま視線を動かす | 手 / 視線に沿って溝や尾根ができる（0.12 s に 1 操作、デスクトップは cursor が 0.75 r 動いてから次） |

**視覚フィードバック:**
- 青い光 = 盛り（左手、またはデスクトップで左ボタン押下中 / 何も押していない時の視線カーソル）
- 赤い光 = 掘削（右手、またはデスクトップで右ボタン押下中の視線カーソル）
- VR では手が地表面に近い時のみ表示、デスクトップでは視線が地形に当たる点（`Cursor Max Dist` 以内）がカーソル

**地形の色分け:**
- 平面: 緑の草地
- 急斜面: 茶色の土
- 深く掘った地下: 灰色の岩石

**Inspectorパラメータ:**

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| Sculpt Radius | 0.3 | スカルプトブラシのサイズ |
| Sculpt Distance | 0.15 | スカルプト発動に必要な手 / カーソルと地表面の距離 |
| Sculpt Cooldown | 0.12秒 | 同じ手の操作間の最小間隔（バッファ溢れ防止） |
| Add Smooth | 0.25 | 丘のSmoothUnionブレンド係数（大きいほど滑らか）毎フレーム material に送るので当たり判定と描画が常に一致 |
| Sub Smooth | 0.15 | 穴のSmoothSubtractionブレンド係数（大きいほど滑らかな縁） |
| Sculpt Capacity | 96 | 保持する操作数 (1-128)、超えると最古を上書き ray の各 step と衝突 sample が全部畳むので GPU / Udon コストのつまみ |
| Cursor Max Dist | 6.0 | デスクトップ: 視線で地形を探す距離 |
| Support | (scene) | プレイヤーに追従する見えないコライダー（generator が `TerrainSupport` を作成、手動なら任意の BoxCollider をここに割当 or `TerrainSupport` と命名） |
| Support Height | 0.2 | その箱の厚み、上面が地表面に置かれる |
| Foot Radius | 0.12 | 足の半幅: 中心と ±x / ±z の 5 点の中で最も高い表面に support を置く、尾根の縁で床がパタつかず足が完全に外れるまで乗れる |
| Log Events | off | 盛る / 掘る / クリック / 持ち上げ / 壁押しごとに `[Terrain] ...` を `Debug.Log` 1 行 — VRChat client の `output_log_*.txt` を grep |

**シェーダーパラメータ:** `Light Direction`（シーンのライトに合わせる）、`Enable Soft Shadow`（丘が地面に接触影を落とす、LOD tier 別 48 / 24 / 12 step）、`Fog Density` レイマーチャは表面の 1 px 以内で step を使い切った ray の最接近点を hit として採用するので丘の輪郭に暗い筋が出ず、AO は hard union を sample するので丘の裾の smooth blend が暗いリングに見えません

**シーン要件:** 地形そのものが床です — TerrainSculpt の world に y = 0 の床コライダーを置かないでください（穴に落ちられなくなります）spawn は地形の少し上（y ≈ 0.5）に サンプルシーン generator はこれを設定済み、volume の Cube は (20, 10, 20) なので穴の深さは 2 m、丘の高さは 8 m まで

#### セットアップ（全インタラクティブサンプル共通）

1. シーンに **Cube** を配置（レイマーチングの描画範囲となるバウンディングボリューム）
2. Cubeを十分な大きさにスケール（例: DeformableWallなら `(12, 8, 12)`、TerrainSculptなら `(20, 10, 20)`）
3. `AliceSDF/Samples/DeformableWall`、`AliceSDF/Samples/Mochi`、または `AliceSDF/Samples/TerrainSculpt` から **マテリアル** を作成
4. CubeのMeshRendererにマテリアルを割り当て
5. 同じGameObjectに対応する `*_Collider.cs` スクリプトをアタッチ
6. VRChatで **Build & Test** — VRの手、デスクトップならマウスでインタラクション

**デスクトップモード:** インタラクティブ sample は全てデスクトップでも操作できます: Mochi — 餅をクリック（Use）して掴み、視点を動かして運び、右クリックで分裂、ボタンを離して落とす TerrainSculpt — 見ている場所に左クリック押しっぱなしで盛る、右クリック押しっぱなしで掘る DeformableWall — 左クリック押しっぱなしで見ている場所を殴る、または壁に突っ込む

**マルチプレイヤー:** Mochi は同期されます（owner 権威の manual sync: 餅の配列が `[UdonSynced]`、owner が重力と合体を回して変更がある間 10 Hz で serialize、掴む / 歩いて押すと掴み・接触ごとに 1 回 owner を取るので、最後に操作した人が状態を動かし、他の人はそれを見て押されます 途中参加者は現在の状態を受信、体の凹みは各自ローカル描画のみ）実質「一度に彫れるのは 1 人」で、2 人が同時に別の餅を持つと相手の餅は掴み直すまで止まって見えます TerrainSculpt も同じ方式で同期されます（スカルプトバッファが `[UdonSynced]`、彫った人がストロークの開始時に owner を取り、変更がある間 10 Hz で serialize、全員が同じ地形の上に立ちます）DeformableWall も同期されます（凹み配列が `[UdonSynced]`、叩いた人がその接触の間 owner を取り、凹みが生きている間 10 Hz で serialize、他の人は packet 間を各自ローカルで回復させるので滑らかに戻る 体の溝は各自ローカル）

### サンプルシーンの自動生成

サンプルをインポートした後、すぐに試せるシーンを自動生成できます。

1. Unityメニュー: **ALICE-SDF > Import All Samples** — 未インポートのサンプルを全部 `Assets/Samples/` に入れます (Package Manager の Samples タブで 1 つずつ Import しても同じ)
2. **ALICE-SDF > Generate Sample Scenes**
3. `Assets/AliceSDF_SampleScenes/` にシーンが生成されます
4. 任意の `SDF_*.unity` を開いて **Play** するだけ

どちらのメニューにもスクリプト / エージェント向けのヘッドレス入口があります (2 回に分けて起動、間でインポートしたスクリプトがコンパイルされる、失敗時は exit code 1):

```
Unity -batchmode -quit -nographics -projectPath <project> -executeMethod AliceSDF.Editor.SampleSceneGenerator.ImportAllSamplesBatch
Unity -batchmode -quit -nographics -projectPath <project> -executeMethod AliceSDF.Editor.SampleSceneGenerator.GenerateAllBatch
```

インポート済みのサンプルを自動検出し、Camera + Light + SDFシェーダー適用済みCube + 情報UIを配置したシーンを生成します。DeformableWall / Mochi / TerrainSculpt については上記セットアップと同じ大きさの Cube に `*_Collider` UdonSharp ビヘイビアも追加されるので、手順 1〜5 は不要です (餅・凹み・地形は Play 時に現れます)。マテリアルも `.mat` として保存されるので、インスペクタからパラメータを変更しながらリアルタイムで確認できます。

---

## クイックスタート

### ALICE-Baker (推奨)

1. **Window > ALICE-SDF Baker**
2. `.asdf.json` をドラッグ＆ドロップ（またはペースト）
3. **Bake!** → Shader + Udon + Prefab が自動生成
4. 生成されたPrefabをシーンにドラッグ＆ドロップ

### 手動セットアップ

Bakerを使わず、手書きでSDFを試したい場合の手順です。

1. **マテリアル作成**: `AliceSDF_Raymarcher.shader` を使ったマテリアルを作成。
2. **Cube配置**: シーンにCubeを置き、上記マテリアルをアタッチ（これが描画エリアになります）。
3. **コライダー設定**: 同じGameObjectに `AliceSDF_Collider.cs` をアタッチ。
4. **数式記述**:
   - シェーダー内の `map(float3 p)` 関数
   - Udonスクリプト内の `Evaluate(Vector3 p)` メソッド
   - **この2つに同じ数式を書くだけです。**

```csharp
// 例: Y=0の床と、半径1.5の球体を結合
public override float Evaluate(Vector3 p) {
    float ground = p.y;
    float sphere = (p - new Vector3(0, 1.5f, 0)).magnitude - 1.5f;
    return Mathf.Min(ground, sphere); // Union
}
```

---

## ファイル構成 (UPM)

```
com.alice.sdf/
├── package.json                     # UPMマニフェスト
├── CHANGELOG.md
├── README.md / README_JP.md
├── AGENTS.md                        # コーディングエージェント向け: 検証可能な導入手順・法則/結線の分離・既知エラー
├── Runtime/
│   ├── AliceSDF.Runtime.asmdef      # Assembly Definition
│   ├── Shaders/
│   │   ├── AliceSDF_Include.cginc   # SDF関数ライブラリ (15プリミティブ + 17演算)
│   │   ├── AliceSDF_LOD.cginc       # Deep Fried 動的LOD
│   │   └── AliceSDF_Raymarcher.shader # メインレイマーチングシェーダー
│   └── Udon/
│       ├── AliceSDF_Math.cs         # ベクトル演算ヘルパー
│       ├── AliceSDF_Primitives.cs   # SDF関数 (C# — HLSL完全ミラー)
│       └── AliceSDF_Collider.cs     # プレイヤー衝突判定 + 押し戻し
├── Editor/
│   ├── AliceSDF.Editor.asmdef       # Editor Assembly Definition
│   ├── AliceSDF_Baker.cs            # Baker v0.3 (Deep Fried)
│   └── SampleSceneGenerator.cs      # メニュー: ALICE-SDF > Import All Samples / Generate Sample Scenes (+ -executeMethod 用 *Batch)
├── Samples~/                        # UPMサンプル (Package Managerからインポート)
│   └── SDF Gallery/
│       ├── SampleBasic/             # 地面 + 球体
│       ├── SampleCosmic/            # 太陽系
│       ├── SampleFractal/           # メンガーのスポンジ迷宮
│       ├── SampleMix/              # Cosmic × Fractal 融合
│       ├── SampleDeformableWall/    # インタラクティブ: 壁を触る→凹む→回復
│       ├── SampleMochi/            # インタラクティブ: 掴む・合体・分裂・巨大化
│       └── SampleTerrainSculpt/   # インタラクティブ: 掘る・積む・穴に落ちる
├── HostTests~/                      # Unity なしで走る検証: Mochi collider と alice_sdf golden の突合 (scripts/vrchat-host-parity.sh)
└── Documentation~/                  # README 用メディア (~ フォルダは Unity が無視)
```

## 対応プリミティブ (53種)

| プリミティブ | HLSL | C# | 数式 |
|-------------|------|----|------|
| Sphere (球) | `sdSphere` | `Sdf.Sphere` | `length(p) - r` |
| Box (箱) | `sdBox` | `Sdf.Box` | Branchless min/max |
| Cylinder (円柱) | `sdCylinder` | `Sdf.Cylinder` | Capped vertical |
| Torus (トーラス) | `sdTorus` | `Sdf.Torus` | XZ ring |
| Plane (平面) | `sdPlane` | `Sdf.Plane` | `dot(p,n) + d` |
| Capsule (カプセル) | `sdCapsule` | `Sdf.Capsule` | Line segment + r |
| Cone (円錐) | `sdCone` | `Sdf.Cone` | Y軸キャップ付き円錐 |
| Ellipsoid (楕円体) | `sdEllipsoid` | `Sdf.Ellipsoid` | バウンド補正近似 |
| HexPrism (六角柱) | `sdHexPrism` | `Sdf.HexPrism` | Z軸六角柱 |
| Triangle (三角形) | `sdTriangle` | `Sdf.Triangle` | 3D三角形（厳密解） |
| Bezier (ベジエ曲線) | `sdBezier` | `Sdf.Bezier` | 二次ベジエ + 半径 |
| RoundedCone (丸錐) | `sdRoundedCone` | `Sdf.RoundedCone` | 滑らかキャップ付き円錐 (r1, r2) |
| Pyramid (四角錐) | `sdPyramid` | `Sdf.Pyramid` | Y軸四角錐 |
| Octahedron (八面体) | `sdOctahedron` | `Sdf.Octahedron` | 正八面体 |
| Link (鎖リンク) | `sdLink` | `Sdf.Link` | チェーンリンク (トーラス + Y伸長) |
| RoundedBox (丸角箱) | `sdRoundedBox` | `Sdf.RoundedBox` | 角丸付きボックス |
| CappedCone (切頂円錐) | `sdCappedCone` | `Sdf.CappedCone` | 円錐台 (2半径 + 高さ) |
| CappedTorus (切頂トーラス) | `sdCappedTorus` | `Sdf.CappedTorus` | トーラス弧セグメント |
| InfiniteCylinder (無限円柱) | — (inline) | `Sdf.InfiniteCylinder` | 無限円柱 (XZ平面) |
| RoundedCylinder (丸角円柱) | `sdRoundedCylinder` | `Sdf.RoundedCylinder` | 角丸付き円柱 |
| TriangularPrism (三角柱) | `sdTriangularPrism` | `Sdf.TriangularPrism` | Z軸三角柱 |
| CutSphere (切断球) | `sdCutSphere` | `Sdf.CutSphere` | 平面切断された球 |
| CutHollowSphere (中空切断球) | `sdCutHollowSphere` | `Sdf.CutHollowSphere` | 中空球の切断 |
| DeathStar (デス・スター) | `sdDeathStar` | `Sdf.DeathStar` | 球体のくり抜き |
| SolidAngle (立体角) | `sdSolidAngle` | `Sdf.SolidAngle` | 3D錐セクター |
| Rhombus (菱形) | `sdRhombus` | `Sdf.Rhombus` | 3D菱形 + 丸め |
| Horseshoe (馬蹄形) | `sdHorseshoe` | `Sdf.Horseshoe` | 馬蹄形 / アーチ |
| Vesica (ヴェシカ) | `sdVesica` | `Sdf.Vesica` | ヴェシカ・ピスキス（レンズ形） |
| InfiniteCone (無限円錐) | `sdInfiniteCone` | `Sdf.InfiniteCone` | 無限円錐 (Y軸) |
| Heart (ハート) | `sdHeart` | `Sdf.Heart` | 3Dハート (回転体) |
| Gyroid (ジャイロイド) | — (inline) | `Sdf.Gyroid` | ジャイロイド極小曲面 |
| Tube (チューブ) | `sdTube` | `Sdf.Tube` | 中空円柱 (外径, 厚さ) |
| Barrel (樽) | `sdBarrel` | `Sdf.Barrel` | 膨らみ付き樽 |
| Diamond (ダイヤモンド) | `sdDiamond` | `Sdf.Diamond` | ダイヤモンド / 双円錐 |
| ChamferedCube (面取り箱) | `sdChamferedCube` | `Sdf.ChamferedCube` | 面取り付きボックス |
| SchwarzP (シュワルツP) | — (inline) | `Sdf.SchwarzP` | シュワルツP極小曲面 |
| Superellipsoid (超楕円体) | — (inline) | `Sdf.Superellipsoid` | 球↔箱モーフ (e1, e2) |
| RoundedX (丸十字) | — (inline) | `Sdf.RoundedX` | 丸みのある十字/X形 |
| Pie (扇形) | `sdPie` | `Sdf.Pie` | セクター / 扇形 |
| Trapezoid (台形) | `sdTrapezoid` | `Sdf.Trapezoid` | 台形プリズム |
| Parallelogram (平行四辺形) | `sdParallelogram` | `Sdf.Parallelogram` | 斜め四角形プリズム |
| Tunnel (トンネル) | `sdTunnel` | `Sdf.Tunnel` | トンネル / アーチ門 |
| UnevenCapsule (非対称カプセル) | `sdUnevenCapsule` | `Sdf.UnevenCapsule` | 2半径カプセル |
| Egg (卵) | `sdEgg` | `Sdf.Egg` | 卵形 (回転体) |
| ArcShape (アーク) | `sdArcShape` | `Sdf.ArcShape` | アーチ / 橋 |
| Moon (三日月) | `sdMoon` | `Sdf.Moon` | 三日月形 |
| CrossShape (十字形) | `sdCrossShape` | `Sdf.CrossShape` | 3D十字 / プラス記号 |
| BlobbyCross (有機十字) | `sdBlobbyCross` | `Sdf.BlobbyCross` | 有機的クロス |
| ParabolaSegment (放物線) | `sdParabolaSegment` | `Sdf.ParabolaSegment` | 放物線アーチ |
| RegularPolygon (正多角形) | `sdRegularPolygon` | `Sdf.RegularPolygon` | N角形柱 |
| StarPolygon (星形) | `sdStarPolygon` | `Sdf.StarPolygon` | 星形多角形柱 |
| Stairs (階段) | `sdStairs` | `Sdf.Stairs` | 階段形状 |
| Helix (螺旋) | `sdHelix` | `Sdf.Helix` | 螺旋チューブ (バネ) |

## 対応演算 (17種)

| 演算 | HLSL | C# (Baker生成後) | 効果 |
|------|------|-------------------|------|
| 結合 | `min(a, b)` | `Mathf.Min(a, b)` | 形状を合成 |
| 交差 | `max(a, b)` | `Mathf.Max(a, b)` | 共通部分 |
| くり抜き | `max(a, -(b))` | `Mathf.Max(a, -(b))` | 形状を削る |
| 滑らか結合 | `opSmoothUnion` | インライン展開 (`inv_k` プリコンピュート) | 滑らかに合成 |
| 滑らか交差 | `opSmoothIntersection` | インライン展開 | 滑らかに交差 |
| 滑らかくり抜き | `opSmoothSubtraction` | インライン展開 | 滑らかに削る |
| 無限繰り返し | `opRepeatInfinite` | `Sdf.RepeatInfinite` | 空間タイリング |
| 有限繰り返し | `opRepeatFinite` | `Sdf.RepeatFinite` | 有限タイリング |
| 極座標繰り返し | `opPolarRepeat` | `Sdf.PolarRepeat` | Y軸円形配列 |
| ねじり | `opTwist` | `Sdf.Twist` | Y軸ねじり |
| 曲げ | `opBend` | `Sdf.Bend` | X軸曲げ |
| 角丸 | `opRound` | `Sdf.Round` | 角を丸める |
| 中空 | `opOnion` | `Sdf.Onion` | 中空シェル |
| テーパー | `opTaper` | `Sdf.Taper` | Y軸先細り |
| ディスプレイスメント | `opDisplacement` | `Sdf.Displacement` | ノイズ表面 |
| 対称 | `opSymmetry` | `Sdf.Symmetry` | 軸ミラー |
| 伸長 | `opElongate` | `Sdf.Elongate` | 軸方向伸長 |

---

## 注意事項と既知の制約

- **GPU負荷**: レイマーチングは画面を大きく覆うとGPU負荷が高まります。`AliceSDF_LOD.cginc` が自動調整しますが、巨大なオブジェクトを配置する際は注意してください。
- **鋭角な衝突**: 「Box」などの鋭い角に対して高速で衝突すると、稀にすり抜ける場合があります（Udonの更新頻度の限界）。`Push Strength` パラメータで調整可能です。
- **VRChat更新**: VRChatの仕様変更により、Udonの挙動が変わる可能性があります。

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| Play 時に `UnityEventFilter` / `Assembly.GetCodeBase` から `Illegal byte sequence` | プロジェクトのパスに非 ASCII 文字 (日本語のユーザー名など) | ASCII のみのパスへ移動して Creator Companion に再登録 |
| `The type or namespace name 'UdonSharp' could not be found` | VRChat Worlds プロジェクトでない、または package を `Assets/` にコピーした | Creator Companion (Worlds) でプロジェクトを作り `Packages/manifest.json` 経由でインストール |
| `[ALICE-SDF] Shader 'AliceSDF/Samples/Mochi' not found` | サンプル未インポート | **ALICE-SDF > Import All Samples** の後にもう一度生成 |
| package を更新したのにサンプルが古いまま | Package Manager は `Assets/Samples/ALICE-SDF for VRChat/<旧バージョン>/` を上書きしない | そのフォルダを削除 → 再インポート → シーン再生成 (再インポートでシェーダーの GUID が変わる) |
| package 解決中に `EPERM`、その後 `UnityEditor.TestTools` のエラーが大量に出る | レジストリ package のダウンロードが rename 途中でロックされ `com.unity.test-framework` が欠けた | `Packages/packages-lock.json` をバックアップ → 壊れた entry を削除 → Unity を前面にして Ctrl+R |
| **Build & Test** が押せない | この PC に VRChat クライアントが無い | クライアントをインストール |
| デスクトップでクリックしても掴めない (`[Mochi] click miss`) | 視線が餅に当たっていない、または Grab Dwell Time より短い押下 | 餅の中心を狙って押し続ける |

grep できる文字列と理由付きの一覧は [`AGENTS.md`](AGENTS.md) §5

## ライセンス

ALICE Community License

## 作者

Moroya Sakamoto

- Website: https://alicelaw.net/
