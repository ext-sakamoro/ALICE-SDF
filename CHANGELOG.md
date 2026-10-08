# Changelog

All notable changes to ALICE-SDF are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

For releases prior to v1.5.0 (v0.1.0 – v1.3.0), see [CHANGELOG-history.md](CHANGELOG-history.md).

## [Unreleased]

### Added
- `examples/bake_assets`: 代表的な 6 つの形 (球、箱、箱から球を引いたもの、smooth union、回転した帯、gyroid と球の交差) について、GPU marching cubes (`gpu-mesh`) と CPU marching cubes を同じ格子で作り、`.abm` (GPU / CPU)・`.glb`・`.nanite`・`.asdf` / `.asdf.json`・compiled bytecode・`manifest.json` (格子、mesh の頂点数と面数と AABB、コライダーのタイトな AABB・体積・重心・慣性テンソル、各 file の SHA-256、crate の版) を書き出し、書き出した file を読み直して検査する (水密、GPU と CPU の mesh の頂点から相手の面までの距離と面積差、頂点の |SDF| ≤ Lipschitz 定数 × 格子幅、球と箱の体積と慣性の閉形式、コライダーの AABB が mesh を含むこと、manifest と mesh の一致、検査した形が 0 件なら失敗) `verify` と、出力を 1 か所ずつ壊す `mutate` も持つ GPU が無い環境では GPU の mesh を飛ばしたことを明示し、`ALICE_SDF_REQUIRE_GPU=1` で失敗にする
- `tests/test_bake_mass_oracle.rs`: bake の質量特性 (Eberly の多面体積分) を手計算の閉形式 (直角四面体、ずらした箱) と marching cubes の球の収束で照合、裏返し・空の入力は `None`
- `scripts/bake_teeth.sh`: bake の出力を壊す 4 通り (GPU の頂点を半格子動かす / 面を 1 枚消す / コライダーの AABB を縮める / 形を 0 件にする) のそれぞれで `verify` が狙った検査で失敗することを確かめる
- `live_sdf` module (`physics` feature): `LiveSdf` は 1 つの形 (基底の `SdfNode` から球の crater を引いたもの + `PhysicsModifier` の列) を共有する handle で、clone はすべて同じ形を指す `SdfField` と `world_participant::Participant` を実装し、同じ handle を `add_sdf_collider` と `add_participant` に渡すと world の step だけで collider の形が modifier に従って変わる (従来は participant と collider の中の modifier が別の object で、step しても collider の形は変わらなかった) 形が変わるたびに世代番号を 1 上げ、変化が届く範囲 (`DirtyRegion`) を記録する snapshot (payload 版 1) で crater と modifier の状態を持ち運ぶ
- `live_sdf::LiveMesh`: `LiveSdf` を chunk 単位の marching cubes で mesh 化し、`sync()` で変化が届く chunk だけを collider と同じ距離関数から再 mesh する chunk の境界の頂点は大域格子の同じ点から canonical な補間で作るので bit 一致し、`merged()` の継ぎ目に穴が無い
- `live_sdf::LiveModifier` (thermal / phase change / pressure / erosion / fracture に実装): 形を決める状態と作用する範囲を返す trait fracture は crack の線分の箱を幅だけ広げた範囲を返す
- `live_sdf::FracturePolicy` と `ImpactContact`: 接近速度が閾値を超えた接触に `destruction_from_impact` と同じ則 (呼び出し) の半径の crater を足す `ImpactContact` の field は ALICE-Physics 2.1 で入る予定の SDF 接触の記録と同じ並び
- `live_sdf::wake_bodies_in`: 変化した範囲にいる休止中の body を起こす (world は休止した body の collider を引き直さないので、形を変えただけでは上に乗った body は動かない)
- `LiveSdf::gpu_mesh` (`physics` + `gpu`): modifier が無い時に `SdfNode` 形を組み立て直して `gpu_marching_cubes` で mesh 化する
- `examples/live_sdf.rs`: 落とした球の衝突で crater ができ、collider と mesh の両方に反映される流れを assert で自己検証する
- `tests/test_live_sdf_oracle.rs` (11 本): crater を掘った後の球の沈み込みが閉形式の半径と一致、world の step だけで collider の距離が `ModifiedSdf` を手で更新した参照と bit 一致し participant 無しでは変わらない、mesh の全頂点の距離が格子幅以下・crater の壁が閉形式の球面上、再 mesh した chunk の集合が閉形式の数え上げと一致し全 chunk が作り直した mesh と bit 一致、継ぎ目の境界辺 0 と除いた体積が半球の閉形式と一致、同じ編集で bit 一致、fracture の範囲、衝突則、snapshot の往復と拒否、不正入力 CI の test job と preflight に追加
- `tests/test_live_sdf_gpu_parity.rs`: GPU marching cubes と CPU の chunk mesh の頂点が双方向に 1e-4 以内で対応 CI の gpu-parity job と preflight に追加
- `tests/test_mc_shared_vertex_oracle.rs`: marching cubes (CPU の `marching_cubes` / `sdf_to_mesh` / `marching_cubes_compiled` / `sdf_to_mesh_compiled`、`gpu-mesh` の `gpu_marching_cubes`) を溶接なしで照合 球・箱・箱から球を引いた形 (種数 5)・回転した帯・格子点が表面ちょうどに来る整数半径の球 (整数格子と半整数格子) で、境界辺 0・非多様体辺 0・連結 1・Euler 標数が閉形式、整数球の頂点数が整数演算で数えた「符号の変わる格子辺」の数、面数が 2V − 4、隣接 cell が同じ格子辺に同じ頂点 (下端から補間した値と bit 一致) を参照、球の法線が `p / |p|` から 1e-3 rad 以内、球と箱の体積が閉形式から 1 % 以内、整数球で同じ位置の頂点の余剰数・組の数と面積 0 の三角形の数が閉形式 (余剰 Σ(k − 1)、組 Σ k(k − 1)/2、面積 0 は 2 Σ(k − 1)、k は表面上の格子点の内側の隣接点数) と一致し位置で併合しても閉じた種数 0、GPU と CPU の頂点数と面数が一致 CI の gpu-parity job と preflight に追加
- `alice_sdf::SEMANTICS_ID`: `alice-det-math` の `SEMANTICS_ID` (32 byte、依存の数値 kernel のどれかが同じ入力で別の bit を返すようになると変わる識別子) の再 export 距離場を焼いた結果や衝突の記録に識別子を付ける利用者が、評価に使った算術を識別子に含められる `tests/test_det_golden.rs` が 16 進で pin するので、`alice-det-math` を上げて算術が変わると CI の全 OS lane で red になる

### Fixed
- 古いままだった version 表記を crate の 5.0.0 に揃えた: Python の wheel (`pyproject.toml` の 0.1.0、今後は `dynamic` で Cargo.toml から取る)、`@alice-sdf/wasm` (0.1.0)、`@alice-sdf/threejs` (1.6.0)、Cinema 4D / Houdini / Maya / Nuke plugin の `__version__` (1.6.0)
- **Behavior change** marching cubes の頂点を格子の辺 (下側の格子点と軸) ごとに 1 つ計算し、その辺を使うすべての三角形が index で参照するようにした (CPU の `marching_cubes` / `marching_cubes_compiled` と `gpu_marching_cubes`) 従来は三角形の角ごとに別の頂点を返し、呼び出し側が位置で溶接していた GPU では隣の cell の同じ辺の頂点が末尾の bit で食い違い、res 64 の球を位置の完全一致で溶接すると境界辺 8,842 本、`deduplicate_vertices` でも 224 本残った 格子点が表面ちょうどにある時は距離で溶接すると非多様体の辺ができた (箱から球を引いた形を余白 3 cell で CPU 555 本 / GPU 642 本) いまは溶接なしで全 scene の境界辺 0・非多様体辺 0 `sdf_to_mesh` / `sdf_to_mesh_compiled` は `deduplicate_vertices` を呼ばない 値が iso と等しい格子点は全経路で外側 (`value < iso` だけが内側) で、その点に集まる辺の頂点は同じ位置の別の頂点として残る (面積 0 の三角形ができうる) そのため `sdf_to_mesh` の頂点数は従来の位置溶接より増える場合がある (例: 整数格子の半径 5 の球 res 16 で 390 → 414、箱から球を引いた形 res 64 余白 3 cell で 22,149 → 23,280) `marching_cubes` の返す頂点数は三角形の角の数 (例: res 64 の球で 92,184) から辺の数 (15,366) に減る
- **Behavior change** `marching_cubes_compiled` / `sdf_to_mesh_compiled` の頂点法線を、局所の辺の端点 e0 の格子勾配から、頂点位置での compiled の中心差分勾配に変えた 従来は同じ頂点に cell ごとに別の法線が付き (res 64 の球で 15,366 頂点中 10,240、箱の角で最大 103°)、法線を含めて重複を除く `sdf_to_mesh_compiled` の球に境界辺 32,528 本が残っていた (CLI の mesh 出力はこの経路) `compute_normals` が false の時の法線は、三角形ごとの面法線から隣接面の面積加重平均に変わる
- **Behavior change** `gpu_marching_cubes` は格子の座標を host で CPU と同じ式で計算して渡し (pass 1 の shader に `axis_coords` binding を足した)、辺ごとの頂点を一度だけ作る pass (2b / 3) と三角形の index を書く pass (4) を足した 頂点の順序は CPU と同じ (格子点、次に軸) で、両者の符号が一致する所では index buffer も一致する 頂点と index の buffer は数えた数で確保する (従来は概算で確保し、超えた分を黙って切り捨てていた) `max_vertices` が 0 でなく頂点数がそれを超える時は `GpuError::BufferMapping` を返す pass 4 は cell 数を 1 次元でなく 3 次元で dispatch する
- `tests/test_mesh_orientation.rs` の「頂点数 == 異なる位置の数」を「すべての頂点が三角形から参照される」に置き換えた (表面ちょうどの格子点では同じ位置の頂点が複数ある、頂点数は上の oracle が閉形式で固定する)
- `ProjectiveTransform` の compiled 評価器 (scalar / `f32x8` / BVH) が、子の距離に点ごとの `min(|1/w|, lipschitz_bound)` でなく定数の `lipschitz_bound` を掛けていた 単位行列以外では tree 評価器と値が違った (非単位行列の 1000 点で 1000 点とも不一致) push 時に lane ごとの補正を計算して pop で掛けるようにし、tree 評価器と bit 一致にした 単位行列 (`w = 1`) の場面は従来と同じ値
- `transforms::projective::projective_transform` が `mul_add` で積和を融合していた crate の他の law と同じく `a * b + c` を 2 回丸める式にした 融合の有無で最終 bit が変わる点があった (同じ 1000 点のうち 186 点) 単位行列では積が正確なので値は変わらない
- `tests/test_projective_unfused_oracle.rs`: 全 entry が 2 進で表せない非単位行列で、tree / compiled scalar / `f32x8` / BVH の 4 経路が独立に書いた融合しない参照式と bit 一致すること、その場面で融合と非融合の結果が 100 点以上で違うこと (歯の確認)

### Changed
- CI: `scripts/version_sync.py` が、crate の version を名乗る file (package.json / uplugin / DCC plugin の `__version__` / 下位の Cargo.toml 等) と、文書の依存行 (`alice-sdf = "X"` / `pip install` / `npm install`) を Cargo.toml の version と突き合わせる file ごとに「追従」か「独立 (理由つき)」を `scripts/version-sync.toml` に登録し、未登録の version 表記は失敗にする
- CI: release の wheel を build するだけでなく、install して `python/tests/smoke.py` を走らせる (aarch64 Linux は x86_64 runner での cross build なので除く) smoke は `version()` と install された distribution の version が Cargo.toml と一致することも確かめる
- CI: `.github/workflows/bake.yml` を追加 (`src/**` などを変える main への push と PR、`v*` tag、手動実行) lavapipe で bake と検査と `bake_teeth.sh` を走らせ、出力を artifact に残す (tag は 90 日、それ以外は 14 日) `scripts/preflight.sh` の full に bake の CPU 部分 (macOS では Metal の GPU 部分も) と `bake_teeth.sh` を足した
- CI: `bake.yml` の push の変更検出を `on.push.paths` (直前の push と比較) から、main で最後に Bake assets が success した commit との比較に変えた (`ci.yml` の changes job と同じ方式) 直前の push と比べると、新しい push に打ち切られた run の変更が bake されないまま残った tag と手動実行は常に bake する
- `examples/bake_assets`: GPU の mesh を距離で溶接せずそのまま使い、格子の余白を 3.5 cell から 3 cell にした GPU と CPU の mesh の頂点数・面数・面積 0 の三角形の数の一致を検査に加え (log に `gpu-vs-cpu: topology GPU … CPU …` を出す)、`bake_teeth.sh` に GPU の mesh に頂点を 1 つ足す変異 (`gpu-extra-vertex`) を足した
- CI: cargo / libtest / cargo-semver-checks の出力を読む parser を、workflow の `CARGO_TERM_COLOR=always` で出る色付き出力で試験する `GpuEvalFuture` doctest の件数は inline の `grep` から `scripts/libtest_count.py` (SGR と `\x1b(B` を除去してから `test result: ok. N passed` を読む、件数不足は `::error::` で理由を出す) に移した escape の除去は `scripts/ansi.py` 1 か所 (CSI / 文字集合の指定 / OSC) にまとめた 現状は libtest の出力が pipe 越しで無着色なので一致していたが、`--color always` で 0 件になることを実測した semver-checks の件数抽出 (`[0-9]+ checks: `) は色付きの実出力で 223 件一致することを `scripts/test_semver_checks_count.py` で固定した `scripts/cargo_output_parser_check.py` は、cargo の出力の文言を読む正規表現を持つ script に印を、その試験に色付き sample を要求し、対象 0 件で fail する ci.yml の python job と `scripts/preflight.sh` (`CARGO_TERM_COLOR=always`) で走らせる
- `alice-det-math` を 0.3.1 から 0.4 に上げた 0.4.0 で bit が変わったのは `atan64` / `atan2_64` (`f64`) だけで、この crate の評価器はどちらも呼ばない (`f32` の `atan` / `atan2` と `simd` 版は係数が別で、0.3.x と同じ実装) `tests/test_det_golden.rs` の node ごとの hash と VRChat sample の golden 7 本 (全 21762 点) は 0.3.2 と 0.4.0 で bit 単位で一致した 変わったのは上の `SEMANTICS_ID` の値だけ
- `scripts/det_math_guard.py`: 対象に `src/transforms/` と `src/types/` を足した (`ProjectiveTransform` の law と `types::Aabb` がここにある) 走査した file が 0 件なら fail する (dir の移動や作業 dir の誤りで素通りしない) `types::Aabb::surface_area` は `mul_add` をやめて `2·(dx·dy + dy·dz + dz·dx)` を 2 回丸めで計算する

## [5.0.0] - 2026-10-09

破壊的変更を含む major release 4.x から上げる時は下の移行方法を参照

**破壊的変更と移行方法** (5.0.0)

- `compiled::GpuEvalFuture` は `GpuEvalFuture<'a>` になり、生ポインタでなく `&'a GpuEvaluator` を持つ (4.x では評価器を drop してから `wait` / `resolve` を呼ぶと safe code で解放済みの memory を読んだ) `eval_batch_submit` の返り値は `GpuEvalFuture<'_>` 移行: 型を書いている箇所は `GpuEvalFuture<'_>` にし、評価器は handle を `wait` / `resolve` で消費した後に drop する handle を評価器より長く (別 thread や `'static` の場所に) 持っていた code は、評価器を `Arc` で共有してその thread で `eval_batch` を呼ぶ形にする `Send` / `Sync` は `unsafe impl` をやめて自動導出になった (`GpuEvaluator: Sync` の時に `Send + Sync`、native の wgpu では従来どおり成り立つ)
- `codec_bridge::SdfVolume::voxel_size` は `f32` から `Vec3` (軸ごとの格子間隔) になった 4.x は 3 軸の最小値を 1 つだけ持ち、格子間隔が軸で違う volume では `world_pos` の y / z がずれた 移行: 構造体を直接作る箇所は `voxel_size: Vec3::splat(step)`、値を読む箇所は `.voxel_size.x` など軸を選ぶ `voxelize_sdf` は軸ごとの `(extent - origin) / (n - 1)` を入れる (立方の格子では 3 成分とも従来の値)
- `codec_bridge` の bitstream に flags bit 3 (軸ごとの格子間隔、header の後に y / z の f32 が続く) を足した encoder は 3 軸の間隔が bit 単位で違う時だけ bit 3 を立てるので、立方の格子の stream は従来と byte 単位で同じ bit 3 の無い stream (4.x が書いたものすべて) は 3 軸とも同じ間隔として読む 4.1.0 以前の decoder は予約 bit を検査しないので、bit 3 の stream を誤読する (bit 2 と同じく前方互換は無い、この版の decoder は bit 4-7 を `DecodeError::UnknownFlags` で拒否する)
- `crispy` module を非公開にした (`alice_sdf::crispy` は無くなる) crate 内で使っていない `fast_recip` / `fast_recip_vec3` / `select_f32` / `branchless_min` / `branchless_max` / `branchless_clamp` / `branchless_abs` / `round_half_up_vec3` / `BitMask64` / `BloomFilter` は削除、使っている `round_half_up` / `fnv1a_hash` / `fast_inv_sqrt` / `fast_normalize_2d` は crate 内専用 移行: 1 行の式なので利用側に写す (`round_half_up(x)` は `(x + 0.5).floor()`、`branchless_min(a, b)` は `a.min(b)` (NaN の扱いだけ違う)、`fast_recip(x)` は `1.0 / x`) Bloom filter と bit mask が要るなら専用の crate を使う
- `compiled::Vec3x8` の未使用 method 12 件 (`zero` / `length` / `length_squared` / `normalize` / `dot` / `abs` / `max_zero` / `max` / `min` / `clamp` / `max_component` / `min_component`) と `compiled::Quatx8::splat` / `inverse` を削除した (型と field、`from_vecs` / `splat` / `new` / `to_array` / 演算子、`Quatx8::mul_vec3` は残す) 移行: field (`x` / `y` / `z` / `w` は `wide::f32x8`) に `wide` の演算を掛ける (例 `v.length()` は `(v.x * v.x + v.y * v.y + v.z * v.z).sqrt()`、`q.inverse()` は `Quatx8 { x: -q.x, y: -q.y, z: -q.z, w: q.w }`)
- `compiled::jit_simd` module と `JitSimd` を削除した (1.9.2 から `#[deprecated]`、`JitSimdSdf` に委ねるだけの wrapper) 移行: `JitSimd::compile` は `compiled::jit::JitSimdSdf::compile`、`eval(px, py, pz, out)` は `eval_8_raw`、`eval_soa` は同名
- `primitives::eval_primitive_unchecked` を削除した (呼び出し元が無く、doc の「compiled evaluator の内側専用」も実態に合わなかった) 移行: 形ごとの `primitives::sdf_*` 関数 (例 `sdf_sphere`) を直接呼ぶ
- `primitives::PrimitiveType` と `primitives::eval_primitive` を削除した (crate 内にも利用側にも呼び出し元が無かった、評価は `SdfNode` と compiled evaluator が担う) 移行: 形ごとの `primitives::sdf_*` 関数を直接呼ぶ
- `mesh::bvh::Aabb` を `types::Aabb` に統合した (同じ `min` / `max` の箱が 2 つあった) `mesh::Aabb` / `mesh::bvh::Aabb` は `types::Aabb` の再 export になり、`types::Aabb` に BVH 側の `empty` / `expand_point` / `expand_aabb` / `surface_area` / `longest_axis` / `signed_distance` を足した 移行: 型の名前を書いている箇所はそのままで compile する 両方の型を受ける impl (trait の impl など) を書いていた場合は 1 つにまとめる `mesh::Aabb` は `Serialize` / `Deserialize` を持つようになった
- `mesh::bvh::Triangle` を `BvhTriangle` に改名した (`mesh::Triangle` は index の 3 つ組で、名前が同じなのに別物だった、`mesh::BvhTriangle` の再 export 名と揃えた) 移行: `mesh::bvh::Triangle` を `mesh::BvhTriangle` にする
- `mesh::primitive_fitting::PrimitiveType` を `FittedPrimitiveKind` に、`FittedPrimitive::primitive_type()` を `kind()` に改名した (評価器の dispatch 用の `primitives::PrimitiveType` と名前が同じで意味が違った) `mesh::PrimitiveType` の再 export も `mesh::FittedPrimitiveKind` になる 移行: 名前を置き換える (variant は同じ)
- `mesh::nanite::NaniteCluster` に field `lod_bounds` (group の LOD 球) / `parent_error` (親 group の誤差、根では `f32::INFINITY`) / `parent_lod_bounds` (親 group の LOD 球、根では `lod_bounds` と同じ) を足した 構造体を literal で作っていた code は 3 field を足す (`generate_nanite_mesh` が作る値は `ClusterGroup::bounds` / `max_error` と同じ) `should_render` の意味が変わる (下の Changed)
- `.nanite` の形式を版 3 にした (`io::nanite::NANITE_VERSION` = 3) cluster ごとに material id の後へ 36 byte (group の LOD 球の中心 x / y / z と半径、親の誤差、親の LOD 球の中心 x / y / z と半径) が入る `.nanite.json` の cluster には `lod_bounds` / `parent_error` (根は `null`) / `parent_lod_bounds` が入る 移行: 読み手は版 3 の配置で読み、版 1 / 2 は拒否するか作り直す
- `codec_bridge::DecodeError` を `#[non_exhaustive]` にし、variant `InvalidDimensions` / `InvalidQuantizer` / `InvalidHistogram` / `CoefficientOverflow` を足した 移行: `DecodeError` を網羅 match している箇所に `_ =>` を足す 4.x の encoder が書いた stream はすべて従来どおり読める (下の Fixed)
- 下の Deprecated の項目は `#[deprecated(since = "5.0.0")]` を付けたので、`-D warnings` (`deny(deprecated)`) で build している利用者は置き換えが要る 各項目に置き換え先を書いた

### Added
- `tests/test_aabb_unified_oracle.rs`: `types::Aabb` に移した method (`surface_area` / `longest_axis` / `signed_distance` / `empty` + `expand_point` / `expand_aabb`) を 2 進で正確な箱と格子点で閉形式と照合、`MeshBvh::bounds` が `types::Aabb` を返すこと
- `src/lib.rs` の `compile_fail` doctest (`cargo test --doc`): 5.0.0 で消した / 非公開にした path (`crispy`、`Vec3x8::zero`、`eval_primitive_unchecked`、`primitives::PrimitiveType`、`eval_primitive`、`mesh::bvh::Triangle`、`primitive_fitting::PrimitiveType`) が crate 外から使えないこと、置き換え先が compile すること
- `tests/test_codec_bridge_oracle.rs`: 格子間隔が 3 軸で違う (どれも 2 進で正確な) volume の `world_pos` が軸ごとの `origin + i · step` と bit 一致し、その点で標本を取っていること、bit 3 の byte 配置と往復、bit 3 の無い stream が 3 軸同じ間隔で読まれること、y / z の途中で切れた stream の拒否
- `tests/test_mesh_cache_model_oracle.rs`: `merge_all` が chunk 座標の順に連結した mesh と bit 一致し、挿入順と cache を変えても同じであること
- `tests/test_cache_correctness.rs`: `max_entries = 0` で 200 件がすべて残ること

- `codec_bridge::try_decode_sdf_volume` と `codec_bridge::DecodeError`: 予約 flag bit が立った stream と途中で切れた stream を `Err` で返す decoder (`decode_sdf_volume` はこれに委ね、`Err` の時は panic する)
- examples `codec_bridge` (`codec`) / `asp_bridge` (`asp`) / `sdf_eval_cache` (`sdf-cache`) / `sim_bridge` (`physics`、`gpu` 併用で GPU MC も) / `mesh_cache`: 各 bridge と mesh cache の使い方を、値の出力と assert 付きで示す
- `tests/test_codec_bridge_oracle.rs` (`codec`): voxel が直接の `eval` と bit 一致、ロスレス設定の往復が固定小数点の値 `round(d·scale)/scale` と bit 一致 (raw / rANS の両経路、i16 を超える値を含む)、範囲内の入力の encoder 出力が形式拡張の前と同じ byte 列、lossy の誤差の上限、統計値の手計算との一致、予約 bit と切れた stream の拒否
- `tests/test_asp_bridge_oracle.rs` (`asp`): I-packet の往復 (serde で直列化して送受信する経路を含む) で場が bit 一致、D-packet が delta byte・参照 sequence・`asdf_len` をそのまま運ぶ、`estimate_packet_size` の閉形式、libasp 既定の `to_bytes` (FlatBuffers) は region を運ばず scene が落ちること
- `tests/test_sdf_eval_cache_oracle.rs` (`sdf-cache`): 格子への量子化 (`round`、0.5 は 0 から遠い側) と「最後に入れた値」の model、容量の上限、hit rate = hit / (hit + miss)
- `tests/test_sim_bridge_oracle.rs` (`physics`): 修飾子なしで `eval_compiled` と bit 一致し法線は中心差分 (刻み 1e-3)、各 `add_*` が名前どおりの修飾子を返した index に入れる、alice-physics 1.4 の一様場での閉形式の offset (凍結による成長・内圧による膨張) が順に掛かる、`gpu_mesh_with_physics` が GPU MC と `attach_physics` の合成であること
- `tests/test_mesh_cache_model_oracle.rs`: `MeshCache` を独立な LRU model と、`ChunkedMeshCache` を FIFO・dirty・`.abm` 永続化の model と、決まった乱数列の 500-600 手で突き合わせる (件数・中身・memory 使用量・dirty 集合・`update_sdf_hash` が返す集合・`merge_all` の三角形の多重集合)
- examples `compiled_bytecode` / `instanced_sdf` / `jit_dynamic` (`jit`): compile した bytecode の逆アセンブル (opcode の役割、点を変えるか、距離を後処理するか、部分木の終わり)、`CompiledSdf` / `CompiledSdfBvh` の node 数・命令数・byte 数・Lipschitz 値、半径を書き換えての `refit_all_from_bytecode`、`AabbPacked::distance_to_point_fast`、`eval_compiled_batch` / `eval_compiled_distance_and_normal` / `eval_gradient_simd`、`Vec3R::round` / `max_element` / `InstancedSdf` の scalar・SIMD・batch・個別の距離と `to_instanced_wgsl` (`gpu`) / `JitCompiledSdfDynamic` と `JitSimdSdfDynamic` の `update_params` (再 compile との一致を assert)
- `tests/test_compiled_bytecode_oracle.rs`: opcode の分類を bytecode VM の挙動 (点を書き換える arm、`PopTransform` で距離を後処理する arm) と照合、`next_instruction_index` を独立に書いた frame の対応付けと照合、`node_count` / `memory_size` / `lipschitz` / `aux_data`、`refit_all_from_bytecode` が新規 compile と同じ箱を返すこと、rounded cone / pyramid / octahedron / hex prism / link の箱の閉形式と標本による包含と tight さ、tube / pipe / tongue の箱の包含、batch と SIMD 勾配の scalar との bit 一致、平面と球の tetrahedral 距離 (`d + e²/|p|`)、`InstancedSdf` と木の union の一致
- `tests/test_jit_dynamic_oracle.rs` (`jit`、CI の JIT parity step と preflight): dynamic JIT の parameter 順が抽出関数と一致すること (corpus 全体)、焼き込み版の JIT・interpreter との一致、`update_params` 後の値が新規 compile と bit 一致すること、parameter 数の違う木を拒むこと
- `tests/test_instanced_wgsl_gpu_parity.rs` (`gpu`、CI の gpu-parity job と preflight): `to_instanced_wgsl` の shader を naga で検証し、compute shader として実行した結果を `InstancedSdf::eval_min` と 4913 点で照合する
- `ShaderLang::NAME` (sealed trait の関連定数、診断 message 用)
- `mesh::PointCloudSdf::try_new`: 点と法線の数の不一致と `k_neighbors = 0` を `MeshInputError` で返す構築関数 (`new` はこれに委ねる)
- `examples/shape_analysis.rs`: 体積・表面積・重心・tension の推定、断面 heatmap と colour map、可変厚の shell (`eval_shell*` / `shell_node`)、offset と嵌め合い公差、`export_step_validated` による「印刷できる形だけを書き出す」を、閉形式との照合付きで示す
- `examples/mesh_formats.rs`: STL ASCII / OBJ / PLY / FBX / USDA の書き出しと読み戻し、glTF JSON / IGES / Nanite / ABM / splat / vox の書き出し、FBX の animation clip から timeline への変換 (`--features openvdb` で dense grid、`--features hlsl` で Nanite の material 関数も)
- `tests/test_shape_analysis_oracle.rs`: 球の体積 4/3·π·r³ と表面積、箱の体積、重心、球の断面 (中心・高さ h の 3 平面・任意平面) の距離と画素数、shell と球殻 (annulus) の距離、offset と半径の加減、嵌め合いの最大違反量、`RaymarchConfig::relaxed` の Lipschitz 値を閉形式と照合する
- `tests/test_io_format_oracle.rs`: 往復で三角形の位置が一致すること (`{}` で書く形式は bit 一致) と、STL ASCII の文法、IGES 5.3 の固定桁と DE / P の相互参照、`.splat` の 32 byte 記録、MagicaVoxel の chunk 配置、`.abm` / `.nanite` の header と file 長、glTF の accessor の min / max と data URI を、format の仕様から独立に読んで照合する
- examples (各 example は値を出力し assert で自己検証する): `animation_timeline` (Track / Timeline / AnimatedSdf / `morph`) / `tree_diff_undo` (`tree_diff` / `apply_patch` / `invert_patch` / `merge_patches`) / `constraint_solver` (`ConstraintSolver` の全拘束と `ParamDependencyIndex::bindings_of`) / `llm_schema_validate` / `sdf_collision` (`sdf_overlap` / `sdf_distance` / `sdf_collide` / `compute_manifold` / `sdf_ccd` / `sdf_closest_point`) / `sdf2d_shapes` (`Sdf2dNode` の全コンストラクタと `eval_2d*`) / `neural_sdf_fit` / `raycast_render` (木・compiled・SIMD の marcher と depth / normal renderer、影と AO) / `raycast_jit` (`jit` feature、JIT の marcher と renderer)
- 閉形式の oracle: `tests/test_animation_oracle.rs` (lerp / step / Hermite / loop / 動く球) / `test_diff_oracle.rs` (往復・undo・merge) / `test_constraint_oracle.rs` (連立の閉形式解、増分更新と再 compile の bit 一致) / `test_llm_schema_oracle.rs` / `test_collision_oracle.rs` (球 2 つの接触・分離・TOI・最近点) / `test_sdf2d_oracle.rs` (円・矩形・線分・環・凸多角形の厳密 SDF、CSG と変換) / `test_neural_mlp_closed_form.rs` (重みを手で書いた NSDF で平面と slab) / `test_raycast_oracle.rs` (ray と球の交点、pinhole camera の全 pixel、影と球状空洞の AO、JIT 分は `jit` feature の CI step で走る)
- examples `primitive_distances` / `blend_laws` / `domain_modifiers` / `point_transforms`: 特化した primitive 関数 (`sdf_capsule_vertical` / `sdf_cylinder_capped` / `sdf_plane_from_points` / `sdf_torus_capped` 等)、n 項 CSG と smooth minimum 群、1 軸の modifier (`modifier_mirror_x` / `modifier_twist_z` / `modifier_repeat_y` / `modifier_repeat_polar` / `ifs_fold` / `fbm_noise_3d` / `sweep_bezier_dist_y` 等)、点の変換と `Aabb` / `Ray::at` / `SdfTree::with_metadata` を使い、値を閉形式と突き合わせて出力する
- `tests/test_primitive_closed_form_oracle.rs` / `test_csg_multi_oracle.rs` / `test_domain_modifier_oracle.rs` / `test_point_transform_oracle.rs`: 上の関数を `f64` の閉形式 (線分距離、円柱の (半径, 軸) 箱則、3 点平面、弧への総当たり距離、平面回転、セル折り返し、IFS の貪欲折り返しの独立実装、fBm の定義、Bezier への総当たり距離) と比べる
- examples: `mesh_queries` (`MeshBvh` / `BvhTriangle` / `mesh::Aabb` の距離と最近点、`MeshSdf` の各 config と `to_sdf_node`、`ExteriorField`、`PointCloudSdf`、Hermite の辺交点) / `mesh_collision_repair` (AABB / bounding sphere / convex hull / `simplify_collision` / `convex_decomposition`、`validate_mesh` と `MeshRepair` の各手順、`compute_quality`、球・箱・円柱・平面の fit と `primitives_to_csg`)
- 閉形式の oracle: `tests/test_mesh_query_oracle.rs` (点と三角形の距離を独立実装の総当たりで突き合わせ、内接多面体の球の SDF を Hausdorff 上界 `max(r − sqrt(r² − R²))` 以内で `|p| − r` と照合、緯度経度の点群の被覆半径で点群 SDF を挟む、平面と球の Hermite 交点と格子全辺の交差数) / `tests/test_mesh_collision_fit_oracle.rs` (箱の体積・外接球、凸包の閉性と体積、球 mesh のオイラー標数 2 と穴あけ・穴埋めでの変化、修復手順ごとの除去数、三角形の面積と aspect 比、解析的な点群からの fit の回復、`FittedPrimitive` の距離と `to_sdf_node` の一致)
- examples `mesh_codecs` / `mesh_reorder` / `mesh_presets`: 球の mesh を crate 独自の varint delta 形式 (`encode_mesh` / `decode_indices` / `decode_positions`) と meshoptimizer v1 の index / vertex codec で圧縮して戻し、octahedral / quaternion / exponential の vertex filter と snorm / unorm / binary16 の量子化を往復させる / triangle を混ぜた球に `optimize_spatial_order` → `optimize_vertex_cache` → `optimize_overdraw` → `optimize_vertex_fetch` を掛けて ACMR / ATVR を出し、strip に変換して戻す / `MarchingCubesConfig::aaa` / `AdaptiveConfig::aaa` / `DualContouringConfig::aaa` と compiled の mesher、`DecimateConfig::conservative` / `aggressive`、lightmap UV (atlas と fast)、`compute_uv_density` を使い、値を閉形式と突き合わせて出力する
- `tests/test_mesh_codec_oracle.rs` / `test_meshopt_filter_oracle.rs` / `test_mesh_quantization_oracle.rs` / `test_mesh_reorder_oracle.rs` / `test_mesh_extract_uv_oracle.rs`: module doc の形式から独立に書いた encoder との byte 一致と全 `u32` / 任意の `f32` bit 列での往復、filter の閉形式の誤差上限 (octahedral 射影と逆射影、四元数の最大成分、共有指数の半刻み)、snorm / unorm の全 code の復号と binary16 の IEEE 754 (全 65536 code と隣接 code の中点、ties-to-even)、LRU / FIFO cache の独立 simulation と ACMR の閉形式、strip の往復で三角形の集合と向きが保たれること、Morton の bit interleave、単一視線での cluster の前後順、球の半径・外向きの面・体積 4π/3・triplanar UV、decimation の三角形数の単調減少と二次誤差上限からの形の誤差、lightmap UV の非重複と UV 面積からの texel 数
- examples `autodiff_curvature` / `soa_batch` / `grid_bounds_optimize`: `Dual` / `Dual3` と `dual3_*` primitive の値と勾配、`eval_with_gradient` / `eval_dual3`、`principal_curvatures` / `gaussian_curvature`、`SoAPoints` を slice・iterator・`push` で作って SoA の batch 評価器 (`eval_compiled_batch_soa` / `_parallel` / `_into`) と手での 8 lane 読み書き、`AlignedVec` / `eval_batch` / `eval_grid` / `eval_grid_with_normals` / `grid_index` / `grid_coords` / `eval::gradient`、`Interval` の述語での箱の分類、`optimize` と `optimization_stats`、`TightAabbConfig::preset_medium` を使い、値を閉形式と突き合わせて出力する
- `tests/test_autodiff_oracle.rs` / `test_soa_oracle.rs` / `test_eval_grid_oracle.rs` / `test_interval_predicate_oracle.rs` / `test_optimize_stats_oracle.rs`: 二重数の積・商・合成則と球・箱・トーラス・平面の解析勾配、球の主曲率 1/r とトーラスの外側赤道 (1/r, 1/(R+r))・内側赤道 (1/r, −1/(R−r))、SoA の全経路がスカラーの `eval_compiled` と bit 一致、格子の配置 (`x + y·res + z·res²`) と `eval` の bit 一致、区間の述語を集合の定義と照合、球の区間値が厳密な値域を含み ulp 程度で tight であること、`optimize` が距離を bit 単位で保つこと (dyadic な定数で厳密、k = 0 の smooth union のみ k の下限 1e-10 の 1/4 以内) と除去 node 数、中型 preset の AABB
- examples `material_library` / `npr_bytecode_program`: material の preset と全 builder・texture slot (UV channel と tiling)・`ParticleMaterial`・`material_lerp`・`MaterialLibrary` の lookup を使って GLB に書き出す / NPR の色の木を builder (`multiply` / `plus` / `scale` / `bloom` / `posterize` / `with_hatch` / `with_speed_lines`) で組み、`compile` → `validate` → `serialize` → `deserialize` で 3 形態の評価が一致することと opcode 表・WGSL 評価器の定数を示し、`NprInput` / `distance_field_outline` / `depth_step_outline` / noise の `with_frequency` を使う `npr_scene_shader` は `with_shading` / `with_outline` / `with_camera` の組み込み shading 版も出力する
- `tests/test_material_oracle.rs`: builder が名前の field だけを doc の clamp で変えること (serde JSON の差分で検査)、preset の metallic と屈折率 (ガラス 1.5、ダイヤモンド 2.42、水 1.33)、`material_lerp` の端点の bit 一致と内部の線形の閉形式、library の id、glTF の `textureInfo.texCoord`
- `tests/test_npr_bytecode_oracle.rs`: opcode ごとの wire 形式を独立に書いた encoder と照合、Rust の tag 定数と emit される WGSL の定数の一致、`opcode_word_count` / `stack_effect` の表、`validate` / `deserialize` / `serialize` の各 error、往復した program と木の評価器の bit 一致、DSL builder の閉形式 `tests/test_npr_primitives_oracle.rs`: `NprInput` の内積、球の上の輪郭線の殻、depth step の閾値、noise の周波数の拡大縮小 (bit 一致) と格子点で 0
- `tests/test_npr_bytecode_gpu_parity.rs` (`gpu` feature、CI の gpu-parity job): emit した WGSL の bytecode 評価器を compute shader で実行し、全 18 opcode と 5 つの palette source の program を CPU の木の評価器と 1024 点で突き合わせる (段差の近くの点は除き、比較が 6 割未満なら失敗)
- examples `svo_octree` (`svo`) / `volume_bake` (`volume`) / `terrain_system` (`terrain,image`) / `gi_cone_trace` (`gi`) / `destruction_carve` (`destruction`) / `texture_reconstruct` (`texture-fit`): SVO の build と点・最近面・ray の問い合わせ、`linearize` と level 表・`as_bytes`・`compact_svo`、`split_into_chunks` と chunk の byte 往復・LRU cache / interpreted・compiled・法線付きの volume bake、trilinear sampling、mip chain、raw と DDS の書き出し、GPU の法線 bake / PNG と生 data の heightmap、洞窟と chamber を引いた `terrain_sdf`、clipmap の mesh、splatmap / 床の上と閉じた空洞の中の hemisphere cone trace、probe grid の bake と probe の読み書き / carve・batch・explode、継ぎ目の voxel の手編集と dirty chunk だけの remesh、debris / texture fit と別解像度での `reconstruct` を、値を閉形式と突き合わせて出力する
- `tests/test_svo_api_oracle.rs` / `test_volume_api_oracle.rs` / `test_terrain_api_oracle.rs` / `test_gi_api_oracle.rs` / `test_destruction_api_oracle.rs`: 各 node が自分の中心の `eval` (compiled は `eval_compiled`) を bit 一致で持つこと、独立に辿った深さで level 表を照合、`#[repr(C)]` の node 配置、切り離した部分木の大きさだけ減る compaction、chunk が node を分割し chunk から組み直した octree が cell 内で同じ値を返すこと、chunk の header と record の配置、独立に書いた LRU の model / 球の解析場との bit 一致と中心差分・四面体差分の法線、mip の footprint の最小、DDS_HEADER / DXT10 の各 field (Microsoft の仕様)、binary16 の最近接偶数丸め (隣の code との比較) / 洞窟の天井の上界 (smooth union の k/4)、角丸の箱の閉形式、ramp の上の clipmap の頂点・法線・面の向き、画素値の線形写像、splatmap の arg-max / probe の軸ごとの範囲判定、probe 中心での sample、閉じた空洞の中の cone の不透明度と hemisphere の閉形式の帯 / chunk の mesh が全体の active cell を 1 度ずつ覆うこと、継ぎ目の voxel を読む chunk の集合、dirty chunk の remesh が全体の remesh と一致すること、全 AABB が格子を覆う時の `max(old, −d₁, −d₂)`、explode の主球が空になり符号の変化が `1.5·r` 以内に収まること (CI の aaa step と gpu-parity の aaa step で走る)
- `examples/lod_nanite_meshlet.rs`: 解像度と decimation の LOD chain (`generate_lod_chain` / `generate_lod_chain_decimated`、各 config の preset と `distance_range` / `resolution_at_level`)、距離・画面誤差での level 選択 (`get_lod` / `select_by_error` / `get_blend_pair` / `LodSelector`)、`ContinuousLod`、`LodChainPersist` の集計 (`summary` / `select_lod` / `total_memory_bytes`)、Nanite の cluster (`clusters_at_lod` / `get_cluster` / `to_mesh` / `total_vertices` と preset)、meshlet (`build_meshlets` / `_scan` / `_adjacency` / `MeshletConfig::quality`)、`ClusterBounds::is_visible` / `screen_error` と `NormalCone` での culling を、値を出力して自己検証する
- `tests/test_lod_nanite_meshlet_oracle.rs`: 各 level の Nanite cluster と meshlet を合わせると、同じ解像度で作り直した Marching Cubes の mesh の三角形の多重集合に一致すること、cluster と meshlet の頂点・三角形の上限、局所 index の範囲、bounds (AABB と全頂点を含む球) と normal cone、V1 meshlet が index 順の貪欲な切り方であること、Nanite の group と DAG を文書化した割り当て規則 (重心の octree cell、level ごとの深さ、最も近い粗い level の包含 cell) から作り直した region と照合 (cell 境界をまたぐ三角形が 1 region にだけ入ること、親の誤差 ≥ 子、親の球が子の球を含むことを含む)、cluster の誤差を単位球への三角形ごとの厳密な片側 Hausdorff 距離 (閉形式) と弦の上限で照合、cut を f64 で評価した選択規則と照合し、葉から根までの各経路でちょうど 1 group が選ばれること、選んだ面積が半径 1 ± e の球の面積の間にあること、閾値を下げると cut が細分され三角形数が減らないこと、遠いほど増えないこと、頂点上限が効いていること (1 region が 256 頂点を超える形で)、LOD chain の各 level と単位球の両側 Hausdorff 距離 (mesh 側は閉形式、球側は点と三角形の総当たり距離) が `max_error` 以内で、`max_error` が mesh 側の厳密な距離 (細かい level までの最大) に 0.2% で一致すること、閾値を満たす level が無い時は `select_by_error` と `LodSelector::select` がともに最も細かい level を返すこと、`LodSelector` の投影誤差の閉形式、`ContinuousLod` が `speed·dt` 以下の歩幅で目標に止まること、視錐台と背面の判定を `f64` の角度の式と照合

- `examples/gpu_eval.rs` (`gpu`): buffer pool と `eval_batch_pooled` / `eval_batch_auto`、`with_workgroup_size` の shader、`new_dynamic` / `update_params` / `eval_batch_full` (GPU の法線) / `extract_params`、`new_async` / `eval_batch_async` / `eval_batch_submit` (`wait` / `resolve`) / `from_shader_async` / `from_wgsl_async`、`transpile_material`、`from_glsl_compute` (`glsl`) を、平行移動した球の閉形式と突き合わせて出力する
- `examples/shader_export.rs` (`glsl,hlsl,blinkscript`): GLSL の compute / fragment / 全 pipeline (`RenderConfig`、`build_full_shader` の `dual_sdf`)、HLSL の compute と Unreal の material function、BlinkScript の body、GLSL / HLSL / BlinkScript の Dynamic parameter layout を書き出す `examples/msl_emit.rs` (`msl`): `MslShader::transpile` / `from_wgsl` と dispatch の metadata、不正な WGSL の `MslError` `examples/rust_transpile.rs` は `RustSource::transpile` / `transpile_with` が `transpile_compiled` と同じ source を返すことも示す
- `tests/test_gpu_eval_api_oracle.rs` (`gpu`、CI の gpu-parity job と preflight): pool の容量の伸び方と再利用、262,144 点を超える batch の分割で点が欠けないこと (300,000 点)、workgroup 64 の shader を全 entry point で、Dynamic の距離・四面体差分の法線・`update_params` 後の値と `eval_batch` / `eval_batch_pooled` / `eval_batch_async`、async の構築と評価、`eval_batch_submit`、GLSL の compute、WGSL の material 関数が最も近い部分木の id を返すことを、平行移動した球の閉形式 (`f64`) と照合する
- `tests/test_glsl_export_oracle.rs` (`glsl,gpu`、CI の gpu-parity job と preflight): `GlslShader::to_compute_shader` (Hardcoded / Dynamic) を naga の GLSL front end で GPU 実行して閉形式と照合、`extract_params` の layout (A の Dynamic shader に B の parameter を渡すと B を評価する)、`to_fragment_shader` と `to_fragment_shader_full` を `RenderConfig` の各 flag 単独・全 flag・`dual_sdf` で naga が parse / validate すること (OpenGL / GLSL ES の宣言だけを Vulkan GLSL に書き換えて検査、書き換えの範囲は file の doc)
- `tests/test_hlsl_export_oracle.rs` (`hlsl,blinkscript`、CI の HLSL step と preflight): `export_ue5_material_function` を C++ compiler で build して `AliceSdf_Eval` / `AliceSdf_Normal` を球の閉形式と照合、`HlslShader::extract_params` / `BlinkScriptShader::extract_params` の layout (Dynamic の emit に別の木の parameter を渡す)、`BlinkScriptShader::get_eval_function`

- `HlslShader::to_unity_custom_function` / `HlslShader::export_unity_shader_graph` (`hlsl`): Unity Shader Graph の Custom Function (Type: File) 用の HLSL file (`void SdfEval_float(float3 p, out float distance)` / `AliceSdf_float` と `AliceSdf_half`)、Dynamic mode の parameter は `float4 _SdfParams[1024]` (`#define params` は file 末尾で `#undef`)
- `GlslShader::to_vulkan_compute_shader` (`glsl`): `to_compute_shader` と同じ kernel を Vulkan GLSL の host interface (`#version 450`、set 0 の binding 0-2、Dynamic の `SdfParams` は binding 3、loose uniform なし) で返す binding 0-2 は `GpuEvaluator` の bind group と同じ
- oracle: `tests/test_hlsl_export_oracle.rs` に UE5 Custom node の body (UE と同じく入力 `p` の関数で包む) と Unity の 2 file を C++ compiler で build して球の閉形式 (距離、Shader Graph 版は法線と half 版も) と照合、Dynamic で A の出力に B の parameter を渡すと B を評価することを追加 `tests/test_hlsl_blinkscript_parity.rs` に同じ 3 出力を corpus 全 node で CPU の評価と照合する sweep を追加 (Custom node は Hardcoded / Dynamic、data array を持つ `lattice_deform` と heightmap を含む) `tests/test_glsl_export_oracle.rs` に Vulkan compute shader を naga で出力のまま検証して GPU で閉形式と照合、`dual_sdf` と multi-material の全 pipeline の naga 検証、`material_slots > 1` の panic、header の version を追加
- `examples/shader_export.rs` が Vulkan compute shader、UE5 Custom node、Unity の 2 file も書き出す

### Changed
- **Breaking:** `physics` feature の `alice-physics` 要求を `1.1` から `2` に上げる (crates.io の 2.0.0 に解決される) SDF 側のコード変更は無く、`cargo test --features physics` は 2116 passed / 0 failed
- 版を 5.0.0 にした (`Cargo.toml`、`unreal-plugin/AliceSDF.uplugin` の `VersionName`)
- `cache::CacheConfig::max_entries` の doc に、`0` は上限なし (何も追い出さない) であることを明記
- `compiled::GpuEvalFuture` に `#[must_use]` と `Debug` を付けた
- CI の test job (3 OS) と `scripts/preflight.sh` に、`cargo test --doc --features gpu GpuEvalFuture` の step を足した
- `io::nanite::NANITE_VERSION` を 1 から 3 に上げる 版 2 (未 release) で cluster の `geometric_error` と親子 id、group 数の意味が変わり (下の Nanite の修正)、版 3 で cluster ごとに LOD 球と親の誤差・LOD 球が入った (冒頭の一覧) 読み手は版を見て 1 / 2 を拒否するか変換する (crate 内に読み込みは無い)
- `examples/lod_nanite_meshlet.rs`: Nanite の cut (`select_clusters` と `should_render`) を閾値と距離を変えて出力し、閾値を下げると三角形数が減らないこと、遠いと増えないことを自己検証する
- **Behavior change:** `mesh::nanite::NaniteCluster::should_render` は cluster 1 つの field だけで標準の cut 全体 (自分の投影誤差 ≤ 閾値 (子が無ければ常に可) で、親の投影誤差 > 閾値 (根は親が粗すぎる扱い)) を判定し、`NaniteMesh::select_clusters` と同じ cluster を選ぶ 距離は cluster 自身の `bounds` でなく group の LOD 球 (`lod_bounds`) まで測る (同じ group の cluster が同じ判定をするため) 4.x と前半だけの版では、描かれる cluster の祖先で閾値を満たすものも `true` を返していた 移行: 前半だけ (細かさの判定) が要るなら `geometric_error / (|lod_bounds.center − 視点| − lod_bounds.radius) ≤ 閾値` を書く `tests/test_lod_nanite_meshlet_oracle.rs` で、各 cluster の 3 field が group と親 group の値と bit 一致すること、`should_render` で選んだ cluster が `select_clusters` と、group の規則を `f64` で評価した選択と一致すること (preview / medium_detail、6 視点 × 7 閾値、閾値 0 と ∞ を含む) を固定 example も per cluster の cut が `select_clusters` と一致することを assert する

- CI: Unreal の workflow は main で走行中の run を打ち切らないようにした (UE 5.7 / 5.8 の直列実行が push の間隔より長く、5.8 が完走しなかった)
- `SimulatedSdf::new` は `SimulatedSdf::from_arc` に委ねる (結果は不変)
- `asp_bridge` の doc の例を、serde で packet を直列化する形に直した (libasp 既定の `to_bytes` は region を運ばないので scene が届かない、旧例は型も合わなかった)
- `codec_bridge` の module doc に bitstream の形式 (header・flags の各 bit・histogram・payload) を書いた
- CI の bridges job と `scripts/preflight.sh` (full) に、上の bridge の 4 file を `--features "physics,codec,asp,sdf-cache,gpu"` で走らせる step を足した
- `refit` の rounded cone / pyramid / octahedron / hex prism / link / ellipsoid / box frame / rect 系の箱を `aabb::primitives` の関数と `AabbPacked::from_half_size` で求める (hex prism 以外の値は不変)
- WGSL / GLSL / HLSL の transpiler が shader の組み立てを `GenericTranspiler::generate_shader` に委ねる (出力は不変) `generate_shader` は helper の source が無い時に panic し、module scope の global (lattice / heightmap の data) を出力する (従来は helper を黙って飛ばし、global を出力していなかった)
- 圧縮した bytecode の評価器と relaxed tracing が `CompiledSdf::instructions` / `aux_data` / `lipschitz` を accessor で読む (結果は不変)
- `extract_jit_params` / `extract_simd_params` は dynamic code generator を scratch の IR 関数に走らせてその parameter buffer を返す (machine code は作らない) 手書きの抽出関数 (約 380 行 + 約 400 行) は削除
- `JitSimdSdf` / `JitSimdSdfDynamic` の private field `module` を `_module` に改名し `#[allow(dead_code)]` を外した (JIT の code を保持するための field)
- `JitSimdSdfDynamic::update_params` の doc に、`Rotate` の quaternion と多項式 smooth 演算の半径は code に焼き込まれ buffer から読まれないことを明記
- `examples/mesh_reorder.rs`: `optimize_overdraw` が三角形の順序を変えることを assert し、`optimize_overdraw_with_views` を上からの 1 視点と `default_view_directions` (逆向きの方向が打ち消し合い順序が変わらない) で呼ぶ (`optimize_overdraw` が両者を呼ばなくなったため、example からの呼び出しで配線する)
- `RaymarchConfig::relaxed` は Lipschitz 値を `fidelity::distance_fidelity(node).safe_step_scale()` から取る (値は従来と同じ: 有限なら `L.max(1)`、上界が無ければ plain tracing)
- 木の評価器 (`eval` / `eval_material` / `eval_gradient`) の `Translate` / `Scale` / `Rotate` が `transform_translate` / `transform_scale` / `transform_rotate` / `transform_rotate_inverse` を呼ぶ (同じ演算、結果は bit 単位で不変)
- `raymarch_with_config` / `raymarch_detailed` の点は `Ray::new` / `Ray::at` で求める、`sdf_collide` / `sdf_distance` / `sdf_overlap` の格子の幅は `Aabb::size` で求める (どちらも結果は不変)
- `collision` (`sdf_overlap` / `sdf_collide`) と `validity` の肉厚判定、`tight_aabb` の区間の判定を `Interval::is_positive` / `is_negative` / `contains(0.0)` で書く (同じ比較、結果は不変)
- `io::obj` / `io::fbx` / `io::usd` / `io::gltf` の書き出しが `MaterialLibrary::get` / `iter` / `len` で material を引く (結果は不変)
- `gi::IrradianceGrid::sample` の probe の読み出しを `get_probe` で行う (補間に使う座標は常に範囲内なので結果は不変)
- CI の aaa integration step を `--features "aaa,image"` にして上の 5 file を加え、gpu-parity の aaa step に `test_volume_api_oracle` を加えた `scripts/preflight.sh` (full) にも同じ 2 step を加えた (aaa の integration oracle は preflight に無かった)

- `volume::gpu_bake::gpu_bake_volume` の shader を `WgslShader::to_volume_shader` で作る (中身が同じ private な複製 `generate_volume_bake_shader` は削除、comment 以外の出力は不変)
- CLI の `bench` の GPU 計測は buffer pool と `eval_batch_auto` を使う (256K 点を超える batch を分割する) GPU の評価が失敗した時は黙って捨てずに表示する
- `GpuEvaluator::eval_batch_submit` の doc を実際の挙動に合わせた (呼んだ時点では何も dispatch せず、`wait` / `resolve` で評価する)
- CI の example の build と `scripts/preflight.sh` に `blinkscript` / `msl` feature を足した preflight の HLSL step で `test_hlsl_blinkscript_parity` も走らせる (CI の HLSL step と対)
- CI: `hlsl-dxc.yml` を足した `HlslShader` の HLSL 出力 6 形式 (`source` / `to_compute_shader` / `export_ue5_material_function` / `to_ue5_custom_node` / `to_unity_custom_function` / `export_unity_shader_graph`) を、全 node の corpus について利用側と同じ形の wrapper に入れて DirectX Shader Compiler (公式 Linux 版 v1.9.2609、SHA-256 固定) で compile する `scripts/preflight.sh` (full) は dxc がある時だけ走らせ、無い時は skip を明示する

### Deprecated

- `GlslShader::to_unity_custom_function` / `GlslShader::export_unity_shader_graph` (`#[deprecated(since = "5.0.0")]`): 出力は GLSL 構文 (`vec3` / `mix` 等) で、HLSL を取る Unity Shader Graph では compile できない `GlslShader` は GLSL の source しか持たないので HLSL を作れない 移行: 同じ木を `HlslShader` で transpile して同名の関数を使う (出力は変えていない)
- `npr::CompiledColorPipeline::fallback_op_count`: すべての `NprColorNode` variant が native opcode に compile されるので常に 0 移行: 呼び出しを消す (opcode 数は `native_op_count`)
- `compiled::SHADER_UNSUPPORTED`: 2.2.0 からすべての node を transpile するので中身 (4 件) が古く、`shader_unsupported_nodes` は常に空を返す 公開型 `[&str; 4]` を変えずに残す 移行: `shader_unsupported_nodes(&node)` を使う
- `primitives::sdf_cylinder_infinite`: `sdf_infinite_cylinder` と bit 単位で同じ重複 `SdfNode::InfiniteCylinder` と他の primitive の命名に合う `sdf_infinite_cylinder` を残す 移行: 名前を置き換える (値は bit 単位で同じ)
- `cache::CacheConfig::disk_cache_dir` / `persist_on_evict`: 実装されておらず、`MeshCache` は disk を読み書きしない (実装はせず非推奨にした) 移行: 構造体は `CacheConfig { max_entries, ..Default::default() }` で作り、永続化は `io::save_abm` / `ChunkedMeshCache::persist_dirty` を使う
- `mesh::PointCloudSdf::new` / `mesh::point_cloud_to_sdf`: 点と法線の数の不一致と `k_neighbors = 0` で panic する 次の major で返り値を `Result<PointCloudSdf, MeshInputError>` にする 移行: `PointCloudSdf::try_new` を使う (同じ場を返す)

### Removed

- `crispy` の未使用項目 10 件、`compiled::Vec3x8` / `Quatx8` の未使用 method 14 件、`compiled::jit_simd` (`JitSimd`)、`primitives::eval_primitive_unchecked`、`mesh::bvh` の `Aabb` の複製 (移行は冒頭の一覧) `docs/MODULES.md` / `docs/API_REFERENCE.md` / `docs/USAGE.md` / `SECURITY.md` の該当記述も消した
- crate 外から到達できない未使用の関数 (`mod` が private で再 export も無い): `smooth_min_exp_rk` / `smooth_min_cubic_rk` / `perlin_noise_3d_batch8` / `modifier_noise_perlin_batch8` / `sdf_torus_oriented` / `rotation_axis_angle` / `rotation_look_at` / `correct_distance_nonuniform`
- `mesh::mesh_to_sdf` の private な未使用関数 `sdf_triangle` (`#[allow(dead_code)]` 付き、`primitives::sdf_triangle` と重複)

- GLSL / HLSL / WGSL の transpiler の private な未使用項目 `FOLD_EPSILON` / `next_var` / `ensure_helper` / `param` (`GenericTranspiler` に移った後に残っていたもの、`#[allow(dead_code)]` を外した)

### Fixed
- **Behavior change:** `tight_aabb::analytic_aabb` / `compute_tight_aabb*`: `Elongate` を含む木で準位集合の bound が `Unsupported` になり、区間探索だけの box が回転で膨らんだ (Z 軸の円柱を `Elongate` → `Rotate` → `Translate` した帯を 45° に傾けると `preset_large` で ±500 近く、格子が帯幅より粗くなり mesh が崩れた) `Elongate` の準位集合は子の準位集合と `[-h, h]` の Minkowski 和に含まれるので、子の領域を `|A|·h` だけ広げる (`A` は木の上で累積した線形写像、平行六面体 `A·[-h, h]` の厳密な AABB) 同じ機会に `Mirror` / `ScaleNonUniform` (正の有限倍率) / `RoundedBox` / `RoundedCylinder` / `SmoothIntersection` / `SmoothSubtraction` (多項式の smooth max は max 以上) / `WithMaterial` を bound に加え、球・円柱・トーラス・カプセルは回転後の box を「回転した box の box」でなく像の厳密な AABB (球は `r·|A の行|`、円盤は `r·hypot(A_i0, A_i2)`、線分は `|A·seg|`) にした bound が締まるので、該当する木の mesh は同じ解像度で格子が細かくなる 区間探索の `Elongate` (`x − clamp(x)` の相関を捨てるので広めに出る) は変えず、真の集合を含むことを固定した `tests/test_tight_aabb_elongate_oracle.rs` に、帯の閉形式 box (9 角度) / 4 点の折れ線 / 回転した丸い葉の閉形式 / `Mirror` が空になる場合 / 乱数木の全固体点の包含 / 区間包含を固定
- **Behavior change:** `codec_bridge::try_decode_sdf_volume` が壊れた stream で panic した (`try_` の約束に反する): quantizer の step / dead zone を検査せず、大きな step で alice-codec の `dequantize` が `i32` を溢れた (overflow 検査ありの build で panic、無しで wrap した値)、逆 wavelet 変換の加算も同様、`width · height · depth` と rANS の histogram を信じたので任意の大きさの確保を求められた 次を `Err` で返す: 寸法の積の overflow と header の symbol 数との不一致 (`InvalidDimensions`)、`step < 1` / `dead_zone < 0` (`InvalidQuantizer`)、histogram の合計が係数の byte 数と違う・1 つの値が半分以上・512 byte 未満・payload の 16 倍を超える symbol 数 (`InvalidHistogram`、encoder はこのどれも書かない)、逆量子化と逆 wavelet 変換で `i32` を出る値 (`CoefficientOverflow`) 逆 wavelet 変換は alice-codec の `Wavelet3D::inverse` と同じ順序と丸めで加算ごとに検査する実装に置き換えた (溢れない入力では bit 一致、unit test で参照実装と照合) 確保は入力長の定数倍に収まる encoder が書く stream の結果は変わらない (既存の往復と byte 列の oracle が通る) `tests/test_codec_bridge_oracle.rs` に、1×1×1 の volume で逆量子化の閉形式 `max(dz,1) + (|q|−1)·step + step/2` が `i32` に収まる境界の両側、2×1×1 / 3×1×1 で CDF 5/3 の逆変換の閉形式と溢れの境界 (後の加算に入らない溢れを含む)、各 field の拒否、encoder の 6 経路 (raw / rANS × i16 / i32、lossy、軸ごとの格子間隔) の stream の全 byte・全 bit の反転と全長の切り詰めで panic しない (切り詰めは必ず `Err`) ことを固定
- **Breaking:** `compiled::GpuEvalFuture` の soundness: lifetime の無い生ポインタを持ち、評価器を drop してから `wait` / `resolve` を呼ぶと safe code で解放済みの memory を読んだ `GpuEvalFuture<'a>` にして評価器を借用させる (移行は冒頭の一覧) drop が compile error になることを `compile_fail` の doctest で固定し、CI の test job と `scripts/preflight.sh` で `--features gpu` の doctest として走らせる (件数 2 未満で fail)
- **Breaking:** `codec_bridge::SdfVolume::world_pos`: 格子間隔が軸で違う volume で y / z の位置がずれた (間隔を 1 つ、3 軸の最小値しか持たなかった) `voxel_size` を `Vec3` にして各軸の間隔を掛ける (移行は冒頭の一覧)
- **Behavior change:** `cache::ChunkedMeshCache::merge_all`: chunk を内部の `HashMap` の順に連結していたので、同じ chunk の集合でも実行ごとに頂点と index の順序が変わった chunk 座標の `(x, y, z)` の昇順で連結する
- **Behavior change:** `mesh::nanite::generate_nanite_mesh` / `NaniteMesh::select_clusters` / `NaniteCluster::should_render`: Nanite のクラスタ LOD の cut が表面を覆っていなかった (球 r=1、`medium_detail` で視点 z=10・閾値 0.01 で 0 個、z=100 では閾値によらず 0 個、z=3 で LOD 2 の 24 個中 8 個) 原因は 3 つ: cluster の誤差が mesh を見ない大きさの近似で、cluster が 1 個の level だけ別の式になり、子より親の誤差が小さくなりえた / `should_render` が「誤差が閾値を超えたら描く」で標準と逆向き / DAG が隣接 level の球の重なりで、1 つの子が誤差の違う複数の親に属した 標準のクラスタ LOD の cut に直す: 各 level を octree cell (重心で割り当て、深さは `min(前の level, round(log2(解像度 / 8)))`) に分けて cell ごとに 1 group とし、group の親は最も近い粗い level の包含 cell の group、cluster の誤差は SDF 上で実測した `|sdf|` の最大 (三角形を重心座標の格子で sample して局所探索で詰める、正確な距離場なら三角形から表面への片側 Hausdorff 距離) を子の group の誤差まで引き上げた値 (親 ≥ 子)、group の球は子の球を含む `select_clusters` は「group の投影誤差 (誤差 / 球の表面までの距離) が閾値以下 (子が無ければ常に可) で、親の group はそうでない」group の cluster を返し、どの視点でも葉から根までの各経路でちょうど 1 group を選ぶ `should_render` は同じ cut を cluster 1 つの field で判定する (Changed の項) 閾値の単位は角度 (rad、画素 `p` は `p · 2 tan(fov/2) / H`) `NaniteMesh::groups` は level ごとに 1 つから region ごとに 1 つになり、`parent_ids` / `child_ids` は region の親子を指す
- **Behavior change:** `mesh::nanite::generate_nanite_mesh`: cluster を三角形数だけで分割していたので頂点数の上限 (`CLUSTER_MAX_VERTICES`) が保証されていなかった 三角形数と頂点数の両方で詰める `curvature_adaptive` は誤差を半分にする代わりに、法線のばらつきが大きい LOD 0 の cluster を 2 つに分ける (誤差は実測のまま)
- **Behavior change:** `mesh::lod::generate_lod_chain` / `generate_lod_chain_decimated`: `max_error` が mesh を見ず bounding box の対角線 / 解像度の半分を返していた (単位球で実際の距離の数十〜100 倍) SDF 上で実測した mesh から表面への距離にし、細かい level の値を下回らないようにする
- **Behavior change:** `mesh::lod::LodChain::select_by_error`: 閾値を満たす level が無い時に最も粗い level を返し、`LodSelector::select` (最も細かい level) と規則が逆だった 品質優先で最も細かい level に揃える (`select_by_error` の comment「Start from highest detail」も code と逆だったので直す)

- **Behavior change:** `HlslShader::to_ue5_custom_node`: Custom node の body に helper と `sdf_eval` の関数定義をそのまま入れていたため、HLSL は関数の入れ子を許さず compile できなかった body の中で local の `struct AliceSdfCustomNode` を定義して helper と `sdf_eval` をその member 関数にし、instance 経由で `sdf_eval(p)` を返す module scope の data array (lattice の制御点、heightmap) はそれを読む member 関数の `static const` local に移す 距離は `export_ue5_material_function` と同じ
- **Behavior change:** `GlslShader::to_fragment_shader_full`: `RenderConfig::material_slots > 1` では transpile した `float sdf_eval` と multi-material の pipeline (`vec2 sdf_eval` と `getMat` を要求) の型が合わず compile できない shader を返していた 生成時に panic する (doc の `# Panics`、transpile した木は material を 1 つしか持たない) multi-material は `render_pipeline::build_full_shader` に自分の source を渡す
- **Behavior change:** `GlslShader::to_fragment_shader_full`: `dual_sdf` では未定義の `sdf_eval_lite` を参照して compile できなかった scene 自身を `sdf_eval_lite` にする
- **Behavior change:** `render_pipeline::build_full_shader`: `struct Mat` が SDF の scene より後 (BRDF の節) で宣言されていたため、`material_slots > 1` で scene の source が定義する `Mat getMat(..)` (COOKBOOK の例) が compile できなかった `struct Mat` を scene の前で宣言する
- 全 pipeline の shader の header が `Generated by alice-sdf v1.4.0` のままだった crate の version (`CARGO_PKG_VERSION`) を書く
- **Behavior change:** `mesh::nanite::ClusterBounds::is_visible`: `dot > fov_cos − cos α` は円錐と球の判定にならず、半径 0 の球が視線から 90° 外れていても半角 30° の視錐で visible になっていた 半角 θ (`fov_cos = cos θ`) の円錐と、`sin α = r/d` で見込む球が交わる条件 `angle ≤ θ + α` で判定する (doc に追記)
- **Behavior change:** `mesh::nanite::NormalCone::is_backface_culled`: `axis · v >= cos β` は β = 45° の時しか正しくなく、開き角 60° の cone で表向きの面がある視線でも cull していた 全 face が裏向きになる `angle(axis, v) <= 90° − β`、つまり `axis · v >= sin β` で判定する (meshoptimizer の `cone_cutoff` と同じ、β > 90° は cull しない)
- **Behavior change:** `mesh::lod::ContinuousLod::update`: 歩幅 `speed·dt` が目標までの差より大きいと目標を越え、前後で振動し続けて止まらなかった 差が歩幅以下なら目標で止まる 空の chain では何もしない
- **Behavior change:** `mesh::meshlet::build_meshlets` / `build_meshlets_scan` / `build_meshlets_adjacency`: `max_vertices` が 1 か 2 の時、上限を超える 3 頂点の meshlet を返していた 三角形 1 枚が入らない設定は doc の「制約違反は空 Vec」に従って空を返す
- **Behavior change:** `codec_bridge::encode_sdf_volume`: 量子化後の係数を常に i16 に丸めて書いていたため、係数が i16 の範囲を超える体積 (ロスレス設定の `EncodeConfig::lossless()` / `high_quality()` では |距離| > 約 8) は黙って切り詰められ、往復で値が変わっていた (半径 1 の球を [-10, 10]³ の 8³ で: 512 voxel 中 440 が不一致、最大誤差 8.0) 係数が 1 つでも i16 を超える時だけ flags の bit 2 を立てて係数を i32 で書く (bitstream 形式の拡張) 範囲内の入力の出力は従来と byte 単位で同じ decoder は bit 2 を読み、bit 3-7 が立った stream を拒む
- **Behavior change:** `cache::ChunkedMeshCache::load_chunk`: 未 cache の chunk を disk から読むと容量を確かめずに追加していたため、chunk 数が `max_cached_chunks` を超えていた `set_chunk` と同じ FIFO で最古の chunk を追い出す
- **Behavior change:** `destruction::MutableVoxelGrid::remesh_chunk`: chunk の高い側の面で 1 cell 手前で止まっていたため、chunk の継ぎ目を跨ぐ cell が mesh されず、chunk の mesh を並べると全ての継ぎ目に隙間があった chunk は下端の voxel が自分の中にある cell を受け持ち、全 chunk の mesh の和が格子全体の Marching Cubes と三角形単位で一致する
- **Behavior change:** `destruction::MutableVoxelGrid` の dirty の印 (`set_distance` / `carve` / `carve_batch` / `explode`): chunk の下端の面にある voxel は下の chunk の最後の cell も読むので、その chunk にも印を付ける (継ぎ目を編集すると隣の chunk の mesh が古いまま残っていた)
- **Behavior change:** `destruction::operations::explode`: 乱数の 48 bit 値 (`state >> 16`) を `u32::MAX` で割っていたため [0, 1] でなく最大 65536 になり、破片の半径が `radius` の数千倍になって 1 回の爆発で格子全体を削っていた crate 内の他の生成器と同じく下位 32 bit を使う 破片の中心は中心から `0.8·radius` 以内、半径は `[0.2, 0.7]·radius` (doc に追記)
- **Behavior change:** `volume::export::export_dds_3d` / `export_dds_3d_distgrad`: DDS_HEADER の flags が `DDSD_PITCH` と圧縮用の `DDSD_LINEARSIZE` を同時に立て、mip chain があっても `DDSD_MIPMAPCOUNT` を立てていなかった (comment は `0x8` を PIXELFORMAT と誤記) caps の `DDSCAPS_COMPLEX | DDSCAPS_MIPMAP` は mip chain がある時だけ立てる R16 の texel は切り捨てでなく IEEE 754 の最近接偶数丸め (`mesh::quantization::half_encode` に委ねる)
- **Behavior change:** `volume::bake::bake_volume_compiled`: `BakeConfig::generate_mips` を無視して常に 1 level を返していた (`bake_volume` と同じく mip chain を作る)
- **Behavior change:** `volume::gpu_bake::gpu_bake_volume_with_normals`: 引数 `gradient_epsilon` を無視して shader 内で 0.001 に固定していた uniform の空き (`bounds_min.w`) で渡す
- **Behavior change:** `svo::SvoStreamingCache::insert`: 既にある id を入れ直すと古い chunk の byte を `memory_used` から引かず、容量いっぱいなら無関係な chunk を追い出していた 古い entry を先に外す
- **Behavior change:** `gi::IrradianceGrid::get_probe` / `get_probe_mut`: 範囲判定が平坦化した index だけで、`x == grid_size[0]` が次の行の probe を返していた 軸ごとに判定して範囲外は `None`
- **Behavior change:** `terrain::Splatmap::dominant_material`: 右端を越えた `x` が次の行の texel を読んでいた 範囲外は `get_weight` と同じく全 layer の重みが 0 なので 0 を返す
- **Behavior change:** `JitCompiledSdfDynamic::update_params` が `RepeatFinite` の回数を code の読む値 (半分) でなく回数そのもので書いており、更新後の評価が新規 compile と一致しなかった `extract_jit_params` を code generator と同じ source にした
- **Behavior change:** `JitSimdSdfDynamic::update_params` (`extract_simd_params`) の parameter の順と個数が code generator と 14 種の opcode (cone / pyramid / rounded cone / smooth・chamfer・stairs の和積差 / rotate / repeat finite / scale と smooth の組み合わせ 等) で食い違っており、更新後の評価が壊れていた 同上
- **Behavior change:** `JitCompiledSdfDynamic::update_params` / `JitSimdSdfDynamic::update_params` は parameter 数が compile 時と違う木を panic で拒む code は parameter を固定 index で読むので、短い buffer は範囲外を読んでいた
- **Behavior change:** `InstancedSdf::to_instanced_wgsl` の shader が instance の回転を逆変換でなく順変換で点に掛けていた (`InstancedSdf::eval_min` と回転のある instance で不一致、GPU 実測で相対誤差 1e-2 程度) 逆回転 (転置) にした
- **Behavior change:** `OpCode::is_post_process` が bytecode VM の後処理する opcode のうち `ScaleNonUniform` / `ProjectiveTransform` / `LatticeDeform` / `IFS` / `Taper` / `HeightmapDisplacement` / `SurfaceRoughness` を偽としていた `OpCode::modifies_point` が `IcosahedralSymmetry` / `IFS` を偽としていた VM の挙動に合わせた
- **Behavior change:** `SdfNode::node_count` が `ExpSmoothUnion` / `ExpSmoothIntersection` / `ExpSmoothSubtraction` を葉として 1 と数えていた (子を数えていなかった)
- **Behavior change:** `CompiledSdfBvh` の箱: hex prism の x 方向を apothem でなく頂点 (`apothem · 2/√3`) までにした / tube を `max(R, h/2)` の立方体でなく半径 `R + t`、高さ `h/2` の箱にした / pipe を和集合でなく `(a ⊕ r) ∩ (b ⊕ r)` にした / tongue に `(a ⊕ ra) ∩ (b ⊕ rb)` を加えた いずれも形の一部が箱の外にはみ出していた (標本で実測)
- **Behavior change:** Godot の `get_instruction_count` が命令数でなく node 数を返していた
- `aabb::primitives::link_aabb` の y 方向が `l/2 + r2` で `l/2 + r1 + r2` に足りなかった (呼び出し元は無かった、`refit` から使うにあたって直した)
- **Behavior change:** `fbm_noise_3d`: octave ごとの seed を `seed.wrapping_add(i)` で求める debug build で `seed` が `u32::MAX - octaves` を超えると加算の overflow で panic していた (release build の値は不変)
- `scripts/scip_reach.py`: CLI (`src/bin/`) を到達性の起点に加えた CLI からだけ使う item (`export_stl` / `export_ply` / `io::get_info` 等) が L0 に数えられていた
- `scripts/scip_reach.py` / `scripts/scip_index.sh`: repo 内で本 crate に依存する crate (`server/` / `mobile/uniffi-wrapper/` / `bindings/openxr/`) を個別に索引し、その `src/` を起点に数える (`examples/` は L1、`tests/` は数えない) fuzz と同様に、索引ごとに本 crate への参照が 0 件なら失敗する
- `scripts/wiring_guard.py`: `*const T` (raw pointer 型) を `const` item と誤認していた 後続の method が free fn 扱いになり、`.as_mut_slice()` 等の呼び出しが数えられなかった
- `scripts/wiring_guard.py`: 同じ file に同名の item が 2 つ以上ある時は key に所有する型を付ける (`src/x.rs::Type::name`) baseline の 1 行が両方を覆い、一方が新たに未配線になっても検出されなかった
- **Behavior change:** `shell::shell_node` が非対称の shell (`inner_offset != outer_offset`) で内側の厚みを外側にも使い、帯を `[r − inner, r + inner]` にしていた `eval_shell` と doc が定める `[r − inner, r + outer]` になるよう `Onion { Round { child, radius: (outer − inner)/2 }, thickness: (outer + inner)/2 }` で組む 対称の場合 (`inner == outer`) は従来と同じ値
- **Behavior change:** `io::iges::export_iges` が P section の 65-72 桁に P の行番号を書いていた IGES 5.3 §2.2.4.5 どおり、その entity の DE の番号 (2n − 1) を書く 出力の byte 列が変わる
- `io::gltf::export_gltf_json` が `buffers[0]` に `uri` を持たない .gltf を書いており、頂点データが file に含まれていなかった buffer を base64 の data URI として埋め込む (glTF 2.0 §3.6.1.1)
- `io::fbx::import_fbx` / `import_fbx_full` がバイナリ FBX を UTF-8 として読もうとして I/O error を返し、「Binary FBX import not supported」の判定に届いていなかった 先に magic を見て `IoError::InvalidFormat` を返す
- **Behavior change:** `animation::morph(from, to, blend)` が doc の通り `(1 − blend)·from + blend·to` (`SdfNode::Morph`) を返す 従来は `k = 1 − blend` の smooth union で、どの blend でも両方の形が残り、blend 0 / 1 でも `from` / `to` にならなかった
- **Behavior change:** `Track::evaluate` が key の時刻ちょうどではその key の値を返す Step 補間では途中の key の時刻で 1 つ前の key の値を返していた
- **Behavior change:** `diff::tree_diff` が、自身の parameter と子の両方が変わった node (例: offset と子が両方変わった `Translate`) を 1 つの `Replace` にする 従来は子の差分だけを出し、`apply_patch` の結果が新しい木にならなかった (parameter の変更が落ちた)
- `diff::apply_patch` の `DiffError::HashMismatch` が不一致の起きた op の path を返す (入れ子の node では常に空の path だった)
- **Behavior change:** `Sdf2dNode::RegularPolygon` の内部の距離を厳密な凸多角形の SDF にした 頂点の近くで `max(dx, dy)` が辺までの距離より浅い値を返していた (外部の値は不変)
- `llm_schema::schema_summary` の node 数を `SdfCategory::count` / `total` から作る 固定値 (計 126、primitive 72、operation 24、modifier 23) が enum (130 / 74 / 25 / 24) とずれていた
- `Sdf2dNode::FontGlyph` の doc を 64x64 grid (4096 値) に直した (32x32 と書かれていた)
- **Behavior change:** `mesh::hermite` の `extract_edge_crossings` / `extract_hermite` / `HermiteExtractor` が、格子の max 側の面 (x / y / z = `resolution`) に乗る辺を走査していなかった 格子は各軸 `resolution + 1` 頂点なので全辺を走査する 面が bounds の端と交わる場合に交点が増える (閉じた形が bounds の内側にある場合は不変)
- **Behavior change:** `FittedPrimitive::to_sdf_node` の `Box` が `SdfNode::box3d` (全幅をとる) に半幅を渡し、`Cylinder` が `SdfNode::cylinder` (全高をとる) に半高を渡しており、どちらも半分の大きさの node を返していた `FittedPrimitive::distance` と同じ大きさになるよう直した (`primitives_to_csg` の結果も変わる)
- **Behavior change:** `mesh::compute_quality` の aspect 比が `4·√3·A / P²` で、正三角形で 1/3 だった doc の「1.0 = equilateral」どおり `12·√3·A / P²` にした (`min_aspect_ratio` / `avg_aspect_ratio` が 3 倍になる)
- **Behavior change:** `mesh::fit_plane` が、点が厳密に同一平面上にある (共分散行列が特異な) 時に法線の初期値 `(1, 1, 1)/√3` をそのまま返していた 特異な場合は共分散の行の外積 (零空間) を法線にする 非特異な場合は不変
- **Behavior change:** `mesh::quantization::half_encode` が binary16 の subnormal 域 (`|v| < 2^-14`) で値の半分の code を返していた (例: `half_decode(2)` を encode すると 1) IEEE 754 の roundTiesToEven で `full >> shift` を丸める 正規数の域は不変
- **Behavior change:** `mesh::meshopt_filter::encode_filter_quat_one` / `encode_filter_quat_i16` が doc の通り入力の四元数を正規化する 復号は最大成分を `sqrt(1 − x² − y² − z²)` で作るので、長さ 1 でない入力は別の回転に戻っていた
- **Behavior change:** `mesh::meshopt_filter::encode_filter_exp_one` の指数の下限を -126 にした 復号は `2^e` を正規数の bit `(e + 127) << 23` で作るので、`e = -127` は 0.0 になり `|v| < 2^(bits − 128)` の値が 0 に戻っていた
- **Behavior change:** `mesh::overdraw::optimize_overdraw` が既定の視点 (±X/±Y/±Z) で三角形の順序を変えていなかった 向きが逆の 2 方向では各 cluster の 2 つの順位の和が全 cluster で等しくなるため meshoptimizer `meshopt_optimizeOverdraw` と同じ方式 (16 entry FIFO の hard / soft 境界で cluster を作り、平均法線 · (cluster 重心 − mesh 重心) の降順に並べる) にした `threshold` は meshoptimizer と同じ「ACMR の悪化の許容率」(1.05 で 5%、0 以下は soft 分割なし) 従来は未使用だった 凹な mesh の overdraw が減る `optimize_overdraw_with_views` は不変 (doc に逆向きの方向が打ち消し合うことを追記)
- `mesh::optimize::compute_atvr` / `optimize_vertex_fetch` の doc が「`optimize_vertex_fetch` で ATVR が改善する」としていた ATVR は頂点の番号付けに依存しない指標で、番号を振り直すだけの `optimize_vertex_fetch` では変わらない doc を事実に合わせた (code は不変)
- **Behavior change:** `mesh::fit_sphere` が最小二乗になっておらず、半球の点群で中心が大きくずれていた (全球なら回復) 代数的な最小二乗 (`|p|² = 2c·p + (r² − |c|²)` の線形最小二乗) に幾何誤差の Gauss–Newton の仕上げを加えた 全球・半球・球冠のいずれでも中心と半径を回復する 同一平面・同一直線上の点は球を決めないので `None` を返す `inlier_threshold` は結果の inlier 数にだけ使う (従来は当てはめから外れ値を除いていた) `detect_primitive` の結果も変わりうる
- **Behavior change:** `mesh::convex_decomposition` が x 方向に離れた 2 つの部分を 1 つの part にまとめていた 内部の判定が x 方向の偶奇の走査だけで、面に乗る行で偶奇が反転して隙間を埋めていた 格子の境界から空の voxel を 6 近傍で flood fill し、届かない voxel を内部とする 離れた部分は軸の向きに依らず別の part になり、part の頂点数も変わりうる
- **Behavior change:** `mesh::PointCloudSdf` の符号が doc の「K 近傍の重み付き投票」でなく最近傍 1 点の法線だけで決まっていた K 近傍が `1 / 距離` の重みで ±1 を投票する方式にした (K = 1 は従来と同じ、距離の大きさは従来どおり最近傍までの距離) `k_neighbors = 0` は評価時の index の範囲外で panic していたので、`PointCloudSdf::try_new` で error を返す (`new` / `point_cloud_to_sdf` は構築時に panic する)
- **Behavior change:** `mesh::EdgeCrossing::t` が、refine した交点 `intersection` でなく端点の距離の線形補間の t を返していた `start + t · (end − start)` が `intersection` になる t を返す (場が辺に沿って線形なら従来と同じ)
- **Behavior change:** `mesh::convex_hull_from_points` / `compute_convex_hull` の `ConvexHull.vertices` に、構築中に凸包の頂点になり後で内部に入った点が残っていた 面が参照する頂点だけを (追加された順に) 残し、index を詰め直す
- `mesh::mesh_codec::encode_indices` / `decode_indices`: index の差分を `wrapping_sub` / `wrapping_add` で取る debug build で差分が `i32` を超える index 列 (例: 0 の次に 2^31) を渡すと overflow で panic していた (release build の byte 列は不変)
- `mesh::stripifier::unstripify_bound(1)` が `index_count - 2` の underflow で panic し、1 index の strip を渡した `unstripify` も panic していた 3 未満は 0 を返す
- **Behavior change:** `autodiff::principal_curvatures` が判別式に `(k1 + k2)² − (k1² + k2²)` を使っており、球でも k1 ≠ k2 (半径 r で 1.707/r と 0.293/r) を返していた (`gaussian_curvature` は 1/r² の半分) 正しい `(k1 − k2)² = 2(k1² + k2²) − (k1 + k2)²` に直し、接平面への射影の Frobenius ノルムに `−2|Hn|²` の項を入れた (勾配の長さが 1 でない場でも形状作用素の固有値になる)
- **Behavior change:** `SoAPoints::push` が padding (`from_vec3_slice` / `ensure_padding` / `FromIterator` が足す 0) の後ろに点を追加しており、`get(len − 1)` と batch 評価器が push した点でなく padding の 0 を読んでいた また padding の無い状態で push した点の最後の端数の lane 群は、`eval_compiled_batch_soa` が原点で評価し、`_into` (と padding 後 256 点以上の `_parallel`) は範囲外の slice で panic していた push は index `len` に書き、配列を常に `padded_len` まで埋める
- `optimize` が入れ子の変換を合成した結果の恒等変換 (`Scale(2)·Scale(0.5)`、打ち消し合う `Translate`) を残していた 合成の後にもう一度恒等変換を畳む (距離は不変)
- **Behavior change:** `npr::compiled_color::emit_wgsl_bytecode_evaluator` が返す WGSL の色の法則が CPU (`npr::toon` / `npr::composition` / `npr::hatch` / `npr::motion`) と `npr::shader_glue` の helper と食い違っていた toon (`floor(t·b)/b` → `min(floor(t·b), b−1)/max(b−1, 1)`)、soft toon (帯の上限と smoothness の下限 0.001)、posterize (`/L` → `/(L−1)` と clamp)、bloom (閾値以下で入力色 → 黒)、vignette、hatch (線の法線でなく線の方向に射影しており、mask も常に 1 だった)、speed line (mask が常に 1、焦点と count 0 で 0) を CPU と同じ式にした GPU で program を評価した色が変わる
- **Behavior change:** `io::gltf` の書き出しが texture slot の `uv_channel` を捨てていた 0 以外なら `textureInfo.texCoord` を書く (glTF 2.0 §5.36)
- **Behavior change:** `StandardMaterials::water` の屈折率が既定値 1.5 のままだった 1.33 にする
- **Behavior change:** `GpuEvaluator::eval_batch_pooled` / `eval_batch_auto` / `eval_batch_async` / `eval_batch_full`: workgroup の数を 256 で割って求めていたため、`WgslShader::with_workgroup_size` で 256 未満にした shader では一部の点しか評価されず残りの出力は 0 のままだった (workgroup 64・1000 点で 744 点) `eval_batch` と同じく shader の workgroup size で割る
- **Behavior change:** `GpuEvaluator::new_dynamic` で作った evaluator の `eval_batch` / `eval_batch_pooled` / `eval_batch_async`: parameter buffer (binding 3) を持たない 3 binding の bind group を使っていたため wgpu の validation error で panic していた `eval_batch_full` と同じ 4 binding の bind group を使う
- **Behavior change:** `GlslShader::to_fragment_shader_full` / `render_pipeline::build_full_shader`: `PI` / `TAU` を `biome_terrain` の時だけ定義していたため、既定の `RenderConfig` (昼夜 cycle が `TAU` を使う) では未定義の識別子で compile できなかった 常に定義する また `#version 300 es` に無い `gl_FragColor` に書いていた (GLSL ES 3.00 で廃止) `out vec4 fragColor;` を宣言してそこに書く

## [4.1.0] - 2026-10-07

### Added

- `rust` feature: `compiled::rust::RustSource::transpile(&SdfNode)` が依存の無い Rust source (`pub fn sdf` / `pub fn normal`) を出力する
  距離は `eval_compiled`、法線は `eval_compiled_normal` と bit 一致、未対応の opcode は `RustTranspileError::UnsupportedOpcode` を返す
- `tight_aabb::analytic_aabb`: 準位集合の bound を木の上で伝播する境界箱 (sphere / box3d / cylinder / torus / capsule と剛体変換・一様 scale・offset・CSG に対応、それ以外は `Unsupported`)
- `scripts/wiring_guard.py`: production から呼ばれない `pub` item と理由の無い `#[allow(dead_code)]` の新規追加で CI を失敗させる (既存分は `scripts/wiring-baseline.txt`)
- `scripts/gen-oracle-status.py` / `scripts/gen-wiring-status.py`: `docs/oracle-status.md` / `docs/wiring-status.md` を生成する (走査 0 件で失敗)
- `scripts/ci_test_coverage_check.py`: feature で切り替わる test file / 分岐を、その feature 付きで実行する CI step が無いと失敗する
- `scripts/docs_lint.py` / `scripts/readme_sync.py`: 公開文書の語彙と CHANGELOG の構造を検査し、README の feature 表・MSRV・使用例と `docs/MODULES.md` を code と突き合わせる
- `docs/MODULES.md` (公開 module の一覧)、`SECURITY.md`、`docs/GETTING_STARTED.md` / `docs/TEXT_TO_3D.md` / `docs/INTEGRATIONS.md` (日本語版あり)
- `scripts/scip_index.sh` / `scripts/scip_reach.py`: rust-analyzer の SCIP 索引で参照を定義に解決し、公開 item を「どこからも届かない (L0)」「example からだけ (L1)」「live」に分ける 新しい L0 と、L0 でなくなった `scripts/integration-baseline.txt` の行で CI を失敗させ、`docs/integration-status.md` を生成する (CI の `scip` job)
- `tests/test_node_backend_matrix.rs`: `SdfNode` の全 variant を木と区間の評価器・compile・JIT (scalar / SIMD)・MSL・Rust 出力に通し、C / Python のコンストラクタの有無と併せて `docs/node-support.md` と突き合わせる 対応が変わったとき、テスト用の入力が無い variant があるとき、shader の dispatch に `_ =>` が入ったときに失敗する

### Changed

- **Behavior change:** `compute_tight_aabb*` は区間探索の結果と `analytic_aabb` の交差を返す 回転した形状の箱が膨らまなくなった (型は不変、`Unsupported` の木は従来と同じ結果)
- README を再構成した: 決定性の範囲と CI での検証環境、Cargo feature 表 (AGPL の crate をリンクする `physics` / `codec` / `sdf-cache` を明示)、検証状況、ライセンスの適用範囲 旧 README の長い節は `docs/` の上記 3 文書へ移した
- crate doc (`src/lib.rs`、docs.rs の表示) を README と同じ使用例と feature 表にした
- `oracle-status.yml` / `wiring-status.yml` は台帳が tree と一致するかを確かめるだけになった (commit しない、`contents: read`) 台帳は変更と同じ commit で再生成する

### Fixed

- `scripts/docs_lint.py` reported an empty `[Unreleased]` (the state right after a release) as "check `categories` compared nothing", so no release could pass the docs gate; an empty section now passes and an `[Unreleased]` with entries but no category heading still fails
- `SdfCategory::count()` / `SdfCategory::total()` が 4.0.0 で追加した variant を数えておらず 72 / 24 / 7 / 23 (計 126) を返していた 正しくは 74 / 25 / 7 / 24 (計 130) 1 variant につき 1 node を作って `category()` で数え直す test を追加し、表がずれると red になるようにした 同じ数を書いていた文書 (ARCHITECTURE / USAGE / bindings・unreal-plugin の README / skills) も更新
- `operations::smooth_min_exp` / `smooth_min_exp_rk` が `|k·a|` が約 88 を超える有限の入力に `±inf` を返していた
- 退化したパラメータで panic していた: `solid_angle` / `pie` / `parabola_segment` / `capsule` / `regular_polygon` の負・NaN の寸法、`noise` の巨大な座標 (debug)、`heightmap_displacement` の幅 0 の map、`surface_roughness` の octaves 32 以上 (24 で打ち切る)
- `io::import_glb_bytes` が破損した GLB (範囲外の参照、整数 overflow、巨大な確保) で panic していた `IoError` を返す
- `raymarch` が退化した field で光線 1 本に数十秒かかっていた ステップ予算に上限 `MAX_STEP_BUDGET` を設けた
- `eval_interval` の `Scale` / `ScaleNonUniform` が係数 0 / NaN / inf で反転した区間を返していた `Interval::entire()` を返す
- CI: feature 付きの integration test 28 本と `test_round_tie_parity` の `jit` / `hlsl` 分岐が実行されていなかった
- CI: push の変更検出が、cancel / red になった run の変更を次の push で取りこぼしていた

## [4.0.0] - 2026-09-30

### Added — `scripts/downstream_check.py` (publish 前の下流影響 gate、local 専用)

`cargo semver-checks` は「この API 変更は破壊的か」に答えるが、⚠️ **「誰が壊れるか」
には答えない** (自 crate の rustdoc JSON しか見ない)。consumer を見る gate は
repo に 1 つも無かった。

依存 repo を走査して宣言を `path only` / `version only` / `version + path` /
`workspace 継承` に分類し、`index.crates.io` の publish 済 version と caret 突合して
**BLOCKED** (その repo は今 `cargo publish` できない) / **UNBLOCKED-BY-PUBLISH** /
**WILL-RECEIVE** / **UNAFFECTED** を出す。`publish = false` の crate は別カウント
(原理的に registry に出ないので blocked / unblocked に混ぜると opt-out が gate を
red にする)。併せて **path を持つ依存 repo の未 push** を
`git ls-remote origin refs/heads/main` と突合する (remote-tracking ref は fetch 時点の
snapshot なので使わない)。

- ⚠️ **CI には配線しない**。sibling checkout は runner に存在せず、この repo から
  再構成できないので、原理的に動かない。`cargo publish` の前に手で走らせる。
- 走査は `os.scandir` の明示 BFS で `target/` 等を降下前に刈る。⚠️
  `Path.glob("*/*/…")` は **出力を filter するだけで降下を止めない**ので、本機の
  335 GB の `target/` に降りて 120 秒で kill された (深さ 6 で prune あり 4.4 s /
  prune なし 150 s 未完)。深さ 3 と深さ 7 で結果が完全一致することを実測したので
  既定は 3、走査した深さと刈った subtree 数を集計行に出す。
- 空振り防止: 依存 0 件 / publish 済 version 0 件 / 分類 0 件 / network 失敗の
  いずれでも fail する。caret 判定は起動時の self-test 48 case が red なら sweep
  自体を止める (⚠️ **実 corpus は caret 上限を 1 度も踏まないので、self-test なしでは
  caret を反転させても出力が 1 文字も変わらない**)。
- 4.0.0 に対する実測: `blocked=0 blocked_publish_false=0 unblocked=5
  unblocked_publish_false=2 will_receive=0 unaffected=4 unresolved=0 unpushed=4
  depth=3` (46 宣言 / 38 crate dir / 29 repo) ⇒ **publish で壊れる下流は 0、
  `alice-lol` / `alice-lol-ui` / `alice-view` / `alice-bamboo` の publish 塞がりが解ける**。

### Changed — semver-checks が major bump 中に 0 件比較で green を返すのを直した

`cargo-semver-checks` は lint を「その lint が要求する bump が既に満たされているか」で
選別するので、⚠️ **Cargo.toml が crates.io より major 先行した時点で全 lint が不要
判定になり 1 本も走らない**:

```
Checking alice-sdf v3.1.1 -> v4.0.0 (major change)
Starting 0 checks, 254 unnecessary on 8 threads
 Summary no semver update required          <- exit 0
```

`no semver update required` は「問題なし」と読めるが、実際は何も比較していない。
この 4.0.0 は **unique 102 件の breaking を 1 件も列挙されないまま** publish 直前まで
来ていた (下記 discriminant 98 件 / `#[non_exhaustive]` 2 件 / struct field 2 件)。

`scripts/preflight.sh` と `.github/workflows/security-audit.yml` の両方で、走った
check 数を読み戻し、**0 または読めない時のみ** `--release-type patch` で 2 本目を
走らせて N>0 を要求する。⚠️ exit 100 は breaking な release の正常な結果なので判定に
使わず、**check 数が 0 でないこと**を hard gate にする (green な run が偽装できない
唯一の量)。1 本目が比較できていれば 2 本目は走らせないので、通常の push に baseline
再 build を課さない。

### Changed (breaking) — `SdfNode` / `OpCode` を `#[non_exhaustive]` に

variant の追加は破壊的変更なので、この 2 つの enum に variant を足すたびに
major が必要だった。4.0.0 で `#[non_exhaustive]` を宣言し、**これ以降の
primitive / opcode 追加を minor で出せる**ようにする。

`#[non_exhaustive]` 自体の影響は **`match` に wildcard arm (`_ => ...`) が
必要になること**。⚠️ ただし同 release で discriminant も動いているので、
この enum に触る呼び出しが見るべき breaking は 2 つある (下の
「discriminant が 98 variant ぶんずれた」を併せて読む)。
enum 単位の `#[non_exhaustive]` が禁じるのは exhaustive match であって
variant の構築ではないので、`SdfNode::Sphere { radius }` のような struct
literal も `SdfNode::sphere(r)` のような constructor も従来どおり使える
(構築を禁じるのは variant 単位に付けた場合)。`alice-lol` は `emit.rs` の
`match` が既に `_ => return None` を持ち、構築側は struct literal と
constructor の併用なのでいずれも影響なし。

同 release で `SdfNode::MetricBall` / `MetricBlend`、`OpCode::MetricBall` /
`MetricBlend` を追加している (下記)。この 2 つが `#[non_exhaustive]` を入れる
きっかけになった最後の「major を要求する variant 追加」。

### Changed (breaking) — `SdfNode` / `OpCode` の discriminant が 98 variant ぶんずれた

上記 2 variant を **末尾でなく `Gyroid` (index 29) の直後に挿した**ため、それ
以降の全 variant の discriminant が **+2** 動いた (`Heart` 30 → 32 …
`Terrain` 127 → 129、98 variant すべて delta は +2)。`#[repr]` を持たない enum
なので言語仕様上 discriminant は不定値だが、**`as isize` / `as u32` で数値化して
serialize / index / FFI に渡している呼び出しは黙って壊れる**。

- 該当は `SdfNode` 98 variant と `OpCode` の同数。`cargo semver-checks` の
  `enum_no_repr_variant_discriminant_changed` が全件を挙げる。
- 依存 30 repo を走査した範囲では **`SdfNode` / `OpCode` を数値化している
  呼び出しは 0 件** (`ALICE-Manga/src/export.rs:181` の
  `mem::discriminant` は同一 build 内での比較なので影響しない)。
- **永続化した discriminant を跨って読む場合は再生成が必要**。値は build 間で
  一致する保証が元から無いので、4.0.0 固有の問題ではなく、ここで初めて実際に
  動いたというだけ。

⚠️ **`#[non_exhaustive]` を入れた後も、variant は末尾に足す**。末尾なら
discriminant は 1 件も動かない。今回の 98 件は挿入位置だけが原因。

### Added — 計量そのものを値にする 2 node と、場の主張を測る 2 API (2026-09-27)

距離場はどれもユークリッドノルムで書かれている `‖p‖₂ − r` が球、
`‖p‖∞ − r` が立方体、`‖p‖₁ − r` が八面体で、**式は同じでノルムだけが違う**
その「ノルム」を parameter にしたのが今回の追加。

- **`SdfNode::MetricBall`** — 重み付きノルム `w₁‖p‖₁ + w₂‖p‖₂ + w∞‖p‖∞` の球。
  重みは検証済みの `alice_det_math::metric::MetricWeights` 経由でしか渡せない
  (負の重みは単位球が凸でなくなり計量でなくなる) `eval_lipschitz` が返す上界は
  **推定でなく閉形式** `√((w₁+w∞)² + 2w₁²) + w₂` で、corpus の差分商 property
  test がそれを実測で裏取りする。立方計量は 1-Lipschitz、八面体計量は √3
  (= 距離を過大申告する = 薄い面を踏み抜く側)。
- **`SdfNode::MetricBlend`** — 2 つの場を球状の皮で混ぜる。`radius` の内側は
  内側の場が **bit 一致で**そのまま、`radius + skin` の外側は外側の場が
  **bit 一致で**そのまま (重みは端で厳密に 0 / 1、補間は端点厳密)。つまり
  バブルを置いても外の世界は 1 bit も動かない。皮の上では勾配に
  `|f_in − f_out| · |∇w|` が乗り、この第 1 項はこの node の性質ではないので
  `eval_lipschitz` は **推測せず INFINITY を返す**。
- **`measure::measure_tension`** — 領域上の差分商の最大と最小を実測する。
  静的上界が「法として何を許すか」を言うのに対し、これは「ここで実際に
  何が起きているか」を言う。`max > 1` は距離の過大申告 (踏み抜き)、
  `min < 1` は緩み (marcher の step 数が増えるだけで正しさは保たれる)。
  皮の張力のように **2 つの法が出会う継ぎ目**は、どちらの静的上界でも
  記述できないのでここでしか測れない。
- **`fidelity::distance_fidelity`** — `eval_lipschitz` の float を
  `NeverOverReports` / `OverReportsBy` / `Unbounded` の 3 値にし、
  `safe_step_scale()` で marcher の除数を直接返す。`L > 1` が踏み抜きを
  意味することは doc にしか書かれていなかったのを型にした。

corpus に 4 node (`metric_ball_cube` / `metric_ball_octahedron` /
`metric_ball_mix` / `metric_blend`) を追加したので、5 経路 parity・Lipschitz
主張・AABB 保守性・区間包含・naga validate・det golden の既存 gate が
そのまま新 node にも効く。

### Changed

- `alice-det-math` 0.2 → 0.3 (`metric` module を使うため、追加のみで既存
  function の bit は不変)


### Fixed — 区間包含と helix の場が壊れていた 3 件 (2026-09-27)

`tests/test_interval_soundness.rs` を足して見つけた。既存の
`interval_eval_contains_point_values` は箱ごとに 8 頂点 + 8 内点を見るので、
**標本の隙間にある破れ**は通り抜けていた。新しい oracle は点の当たり外れに
依存しない 2 つの性質を見る:

1. **八分木の包含単調性** — 箱 `B` を 8 分割した子 `B'` について
   `eval_interval(B') ⊆ eval_interval(B)` が成り立たなければ、証拠点を
   探すまでもなくどちらかが誤り。これは `alice-lol` の判定器が実際に行う
   細分 (`box_children` + `BALL_PROBE_DEPTH`) そのものなので、破れは判定器の
   hot path の破れになる。
2. **勾配に沿った敵対的な極値探索** — 箱の中を `eval_gradient` で下り / 上り、
   見つけた極値が区間に入っているかを見る。無作為標本が見逃す witness を
   狙って探す。

修正した破れ:

- **`PolarRepeat` の扇形が `count ≤ 2` で潰れていた** — 折り返し後の z 幅を
  常に `max_r · sin(ha)` としていたため、`ha ≥ π/2` (count 1 / 2) で
  `sin π ≈ 0` となり z が 1 点に縮み、x の下限も 0 のままだった。`alice-lol`
  側ではこれが **「重なりなし」の偽の証明**になっていた (`flange_mount` の
  cell で区間 `[0.05, 1.10]`、実際の場は `−0.71`)。`ha ≥ π/2` では
  `x, z ∈ [−max_r, max_r]` に切り替える。
- **`ScaleNonUniform` の区間が負値で上下反転していた** — `lo` に最小係数、
  `hi` に最大係数を掛けていたので、子の区間が全て負だと `lo > hi` になる
  (例 `[−0.5, −0.25]` × 0.8 / 2.5 → `[−0.4, −0.625]`)。`eval` は最小係数
  1 つを掛けるだけなので、区間もその 1 係数でスケールする (符号も処理)。
  debug build では `Interval::new` の assert が発火し、release では `hi` が
  本来包むべき値より下にある区間がそのまま流れていた。
- **`Rhombus` の外接球半径が `round_radius` を落としていた** — 丸めの分だけ
  形は大きいので、包含が丸め半径ぶん狭かった。

同時に `Helix` の場そのものを修正した: 曲線までの二乗距離
`r² + R² − 2rR·cos` が曲線上で f32 の桁落ちにより微小な負になり、`sqrt` が
NaN、`NaN.max(d_cap)` が **cap 側の距離を返していた** (曲線上で `−0.309`、
tube の法では `−0.0999`)。GPU の `max` は NaN をそう扱う保証がないので、
CPU / GPU 一致の観点でも危うい。WGSL / GLSL / HLSL の helper も同じ形に
揃えた (演算順序も CPU に合わせた)。

corpus に退行 test を追加 (`polar_repeat_1` / `polar_repeat_2` /
`scale_xyz_interior`) — 修正前の実装に戻すと新 oracle が落ちることを確認済み。
距離が動くのは helix の曲線上のみで、`test_det_golden` の既存 entry は不変
(新 3 entry のみ pin を追加)。

### Changed — 区間演算を外側丸めにした (2026-09-28)

Rust に丸めモード制御が無いので、round-to-nearest の結果を 1 ulp 外側にずらす
形で外側丸めを実装した (単一 IEEE 演算は厳密値の半 ulp 以内なので健全)。
`next_down` / `next_up` は MSRV の都合で bit 操作の自前実装。

- 値を**計算する** op を全て外側丸めに: `sqrt` / `sqr` / `expand` / `Add` /
  `Sub` / `Mul` (interval × interval, × f32) / `length` / `length_xz`
- `Interval::new` 自体を `Interval::outward` 経由にしたので、未監査の call site
  も「1 ulp 緩む」側に倒れる (soundness を落とさない default)
- 値を**選択するだけ**の op (`min` / `max` / `hull` / `clamp` / `abs` / `neg` /
  `intersect`) は厳密のまま

併せて `stairs` の 3 arm が「(a, b) 空間で 1-Lipschitz」と仮定して中心標本 ±
半対角で包んでいたのを撤去した (`glsl_mod` は周期境界で跳ぶので仮定が成立しない)。

`test_det_golden` の pin は不変、`alice-lol` の判定器の未決定率も変化なし。

### Added — 外側丸めを迂回できなくする guard (`scripts/interval_outward_guard.py`)

上の健全性は「値を計算する arm は全て `Interval::new` / `Interval::outward` を
通る」という不変条件に依存していて、それを機械が見ていなかった。新しい arm を
`Interval { lo, hi }` と書けば widening を素通りするが、包含 oracle は点を標本
するので**標本していない隅で 1 ulp 足りない区間**を見つけられず 0.000e0 のまま
になる。

そこで `src/` 内の raw literal を検出し、許す場所には理由を書かせる:

```rust
// ALLOW-RAW-INTERVAL: <なぜこの境界は広げなくてよいか>
```

を literal の開始行か直前 3 行に置く。現状の許可は 13 箇所 (選択のみの op と、
`point` / `ZERO` / `EVERYTHING` / `sqr` の 0 下界のように構成上厳密なもの)。
理由が 12 字未満のものと、literal が無くなったのに残った marker も fail にする
(allowlist が、正当化している code から離れていくのを防ぐ)。

`scripts/preflight.sh` と `security-audit.yml` の両方に入れた。`scripts/**` が
workflow の path filter に無く guard を直しても CI が再実行されない状態だったので
併せて追加した。guard が壊れた時に落ちることは、raw literal / 短い理由 / 残留
marker の 3 通りを実際に仕込んで red を確認している。

### Added — 第三者表記 (`THIRD-PARTY-NOTICES.md`)

距離関数の**数式の形**の出典 (Inigo Quilez 34 file / Mercury hg_sdf 15 file /
Perlin の勾配表) を 1 箇所に集約し、README / README_JP から参照。実装は本
repo 独自の Rust code である旨と、上流のライセンス条項は本 repo 内で未検証
である旨を明記した。

### Added — printability validation (`validity` module)

- `validity` module を新設。「この形状は印刷できるか」に対して **どう確かめたかを
  明示して**答える。`Undecided` を合格に丸めないのが設計の中心。
  - **A: erosion による大域証明** — `prove_erosion` が `Round { radius: -t/2 }`
    (SDF の offset なので厳密) で削った形状を octree + `eval_interval` で調べ、
    三値 `ErosionVerdict` を返す: 削った後も残る材料の witness が取れれば
    `HasThickEnoughRegion`、bound 全体で残らないと証明できれば `EntirelyTooThin`、
    深さが尽きたら `Undecided` (**合格ではない**)。
  - **B: 三角形ごとの局所厚さ実測** — `local_thickness` が重心から `-n` に
    sphere tracing する。内側では `-d(p)` が最近表面までの厳密な距離なので
    step が行き過ぎず、固体を抜けた距離がその点の肉厚になる。1 点 1 点は厳密で、
    標本になるのは *どこで測るか* (= mesh の細かさ) だけ。
  - **overhang は閉形式** — `overhang_stats` が `asin(-n · b)` を返す
    (垂直壁 0 / 水平な下向き面 π/2 / 上向きは負にせず 0)。接地判定は持たないので
    「サポートが要るか」は呼び手の方針に委ねる (箱の底面は π/2 になる)。
  - `validate_for_printing` が A + B + overhang + `MeshValidation` を束ね、
    `ValidityReport::is_printable` は **A が証明であること**も要求する。
  - `export_step_validated` は検証を通らなければ **file を書かず**
    `ValidatedExportError::NotPrintable` を返す。既存の `io::step::export_step` は
    検証しないので、印刷に回す形状はこちらを通す (4.0.0 で既存 API 側を
    validated 経由に統合するかは別途判断)。
  - oracle 12 本 (`tests/test_validity_oracle.rs`) は閉形式のみから期待値を作った:
    板厚 2.0 / 球殻 0.5 / 斜面 π/4 / 水平面 π/2 / 厚板は `HasThickEnoughRegion` /
    0.5 mm 板は `EntirelyTooThin` / 深さ 0 は `Undecided` かつ不合格 /
    薄板の export は拒否されて file が残らない。実装前に red を確認してから書いた。
  - **採らなかった設計を doc に残した**: 「箱が形状の内部と証明され、かつ erode 後の
    外と証明されたら肉厚 < t」は **あらゆる形状で成立してしまう**ので誤り (表面から
    t/2 以内の材料は、後ろがどれだけ厚くても必ず削れる)。局所肉厚は medial axis の
    量で、判定には連結性解析が要る — `alice_lol::law` が `Continuity` を
    「格子解像度依存」として sound 化対象外にしたのと同じ壁。

### Fixed — Unreal CI の engine 存在判定が DL 途中を「ある」と誤認していた

- `scripts/unreal-ue5-ci.ps1` は `Engine\Build\BatchFiles\RunUAT.bat` 1 個の有無で
  engine の有無を決めていたが、`Engine\Build\BatchFiles` は Epic Launcher の DL の
  早い段階で展開される。2026-09-27 に **DL 途中 29.5 GB で `RunUAT.bat` だけが存在する
  状態**を実機で実測した。この状態を通すと UBT / shader compile の段で真因の分かり
  にくい red になる。**size 閾値では判定できない**ことも同時に分かっている: その
  0.4 GB 後 (29.9 GB) には `UnrealEditor.exe` が出現して判定を通過し、install の総量は
  Launcher の「エンジンのみ」構成で 26-30 GB (実測 5.7.4 = 26.4 GB / 5.8.3 = 29.9 GB)
  しかない。判定は file の存在で行う。`Build.version` /
  `UnrealEditor.exe` / `UnrealEditor-Cmd.exe` を加えた 4 点判定にして、不足分を列挙
  した `engine at <root> is absent or incomplete (missing: …)` で弾くようにした
  (editor 本体は DL 後半に来るので完了の代理指標になる)

### Fixed — STEP export が STEP として成立していなかった

- `io::step` は `CARTESIAN_POINT` + `POLY_LOOP` + `FACE_OUTER_BOUND` を並べるだけで
  `CLOSED_SHELL` / `MANIFOLD_SOLID_BREP` / 形状表現 / 単位系を持たず、さらに存在しない
  `#0` を 2 箇所から参照していた (= どの CAD でも開けない)。AP214
  (`AUTOMOTIVE_DESIGN`) の faceted BREP として書き直した: 三角形 1 枚 = `PLANE` 上の
  `ADVANCED_FACE`、境界は `ORIENTED_EDGE`/`EDGE_CURVE` の `EDGE_LOOP` (無向エッジ 1 本を
  隣接 2 面で共有)、`CLOSED_SHELL` → `MANIFOLD_SOLID_BREP` →
  `ADVANCED_BREP_SHAPE_REPRESENTATION` → `SHAPE_DEFINITION_REPRESENTATION`、単位は
  mm (`LENGTH_UNIT`)。tessellate 後に `MeshRepair::repair_all` を通して退化 facet を除去
  (退化 facet は平面を持てず、落とすと shell が開く)。README / README_JP の記述も更新。
  面は依然として平面なので、球は `SPHERICAL_SURFACE` ではなく三角形分割として届く
  (曲面の Phase 2 化は別途)。

### Added — 箱は tessellate せず 6 面で厳密に出す (三相原理 Phase 2)

- 原点中心の軸平行箱 (`SdfNode::Box3d`) は面が厳密に平面なので、Marching Cubes を
  通さず 4 角形 6 枚の `ADVANCED_FACE` として書く。voxel 解像度に関係なく寸法が
  厳密になる (`resolution` の値に依らず体積が `w·h·d` と 1e-6 以内で一致)。
  寸法が機能そのものの部品 (治具 / 嵌合部) を CAD に渡す経路が tessellation 誤差から
  外れる。`StepConfig` に field は追加していない (semver 互換を保つため自動判定)。
  球 / 円柱の解析曲面 (`SPHERICAL_SURFACE` / `CYLINDRICAL_SURFACE`) は seam の扱いが
  reader 依存で壊れやすいので今回は入れていない。

### Added — STEP の独立読み戻し oracle

- `tests/test_step_export_oracle.rs` 4 件。書き出した file を実装を通さずに Part21
  として読み戻し、(1) 全 `#id` 参照が解決する (2) AP214 必須 root 15 種が揃っている
  (3) `CLOSED_SHELL` の面を loop から復元した符号付き体積が `4/3 π r³` と 5% 以内
  (4) 各エッジがちょうど 2 回使われる (閉じている) を検証する。complex entity instance
  (`(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))`) も解釈する。
- `io::step` の既存 unit test を更新: `CARTESIAN_POINT` 数 = 頂点数 + 形状表現の原点 1、
  `VERTEX_POINT` 数 = 頂点数 (位相頂点は mesh 頂点と 1:1)。

### Added — MSL (Metal Shading Language) emit, verified on a real Metal device

- `compiled::msl::MslShader` (feature `msl`, implies `gpu`) turns an `SdfNode`
  into a Metal compute kernel. **MSL is derived, not hand-written**: the WGSL
  emit — already checked against the CPU for every corpus node by the GPU parity
  oracle — is the single source of law and `naga` (`wgsl-in` → `msl-out`)
  translates it. A fourth `ShaderLang` implementation would have meant a fourth
  hand-maintained copy of the 56 `helper_source` laws, which is the drift this
  crate keeps paying for (the mirrored taper that survived in all three
  transpilers until the GPU oracle caught it, `Terrain` emitting GLSL syntax from
  the language-neutral walker, `Elongate` disagreeing with the CPU). `ShaderLang`
  stays sealed at three implementations.
- `tests/test_msl_metal_oracle.rs` (macOS, feature `msl`) is the oracle: every
  corpus node's MSL is compiled by the **Metal runtime compiler**
  (`MTLDevice::newLibraryWithSource`, so no Xcode Metal Toolchain component is
  required) and executed as a compute kernel, then compared with the CPU law on
  the same 2048-point LCG set and the same `1e-4 · max(|d|, 1)` relative
  tolerance the WGSL oracle uses. Measured on an M3: **145 / 145 corpus nodes
  compile and match, worst drift 8.744e-5 (`lattice_deform`)**.
  `ALICE_SDF_REQUIRE_METAL=1` turns "no Metal device" from a skip into a failure,
  mirroring `ALICE_SDF_REQUIRE_GPU`.
- Binding contract: WGSL `@group(0) @binding(N)` maps to Metal `buffer(N)`, and
  `MslShader::sizes_buffer_slot` (one past the highest binding) must hold the
  byte lengths naga needs for the runtime-sized arrays. `fake_missing_bindings`
  is forced off so an unmapped binding is an error instead of silently valid MSL
  that reads the wrong buffer.
- `MslShader::entry_point` is naga's **renamed** function, not the WGSL `main`
  (`main` is reserved in MSL). Using the WGSL name made `getFunction` fail with
  "Function 'main' does not exist" — found by the oracle, which is the point of
  running the kernel rather than only compiling it.


### Added — Unreal Engine compatibility gate (CI builds and tests the plugin on UE 5.7)

- `unreal-ue5` CI job (self-hosted Windows runner, label `ue5`): `RunUAT
  BuildPlugin` compiles and links `unreal-plugin/` for the editor and game
  targets, the editor compiles every `.usf` / `.ush`, and
  `Automation RunTests AliceSDF.Unreal` runs two new automation tests on the
  real DX12 device — `AliceSDF.Unreal.FfiCorpusParity` (every corpus node
  loaded from `.asdf` through `alice_sdf.dll`, `alice_sdf_eval` /
  `eval_compiled` / `eval_compiled_batch` bit-exact against the Rust tree
  evaluator on the `test_det_golden` grid) and `AliceSDF.Unreal.HlslGpuOracle`
  (every corpus node's transpiled HLSL executed as a compute shader and
  compared with the DLL at `1e-4 · max(|d|, 1)` — the crate's first HLSL
  *execution* oracle; naga has no HLSL front end). Script:
  `scripts/unreal-ue5-ci.ps1`, inputs from `examples/unreal_corpus_oracle.rs`
  (`Shaders/CorpusOracle/Corpus/*.ush`, `CorpusOracle/SdfCorpusOracle.usf` and
  `Private/Generated/AliceSdfCorpusManifest.h` are committed and drift-gated).
- `unreal-abi` CI job (GitHub-hosted, `scripts/unreal-abi-check.sh`): the
  shipped `ThirdParty/AliceSDF/include/alice_sdf.h` must equal
  `include/alice_sdf.h`, every header prototype must be exported by the
  cdylib built with `--features unreal` and vice versa, `.uplugin` metadata
  (`VersionName` == crate version, `EngineVersion` 5.x), shader includes
  resolve, generated corpus files current. `docs/UNREAL_ENGINE.md` § CI.

### Fixed — `alice_sdf_version()` reported 1.1.0 since 2.0

- `VersionInfo::current()` held hand-written constants that nobody updated:
  every C caller (the UE5 plugin, the Unity bindings, any FFI user) was told
  the library was 1.1.0 through all of 2.x and 3.0 / 3.1. It now comes from
  `CARGO_PKG_VERSION_*`, and `src/ffi/info.rs` has the two tests that were
  missing — the version had no test at all, which is why the lie kept.
- Found by the UE5 plugin's new startup check (below), on its first run.

### Added — UE5 plugin: the native library is no longer committed, and its version is checked

- `unreal-plugin/ThirdParty/AliceSDF/lib/**` holds only `README.txt` now. The
  `alice_sdf.dll` / `.lib` / `.dylib` in the repository were from 1.7.2
  (2026-02-28): CI always staged a fresh build, so nothing failed, while
  anyone building the plugin from a clone linked against a seven-month-old
  library. `scripts/unreal-abi-check.sh` rejects a committed binary, and the
  release zip ships the real ones.
- `FAliceSdfModule` logs the library's version at startup and reports an error
  when its major / minor differs from the `.uplugin` `VersionName`
  (`AliceSDF.Build.cs` passes that through as `ALICE_SDF_EXPECTED_VERSION`):
  `LogTemp: ALICE-SDF: native library 3.1.0 loaded from ...`.
- `scripts/unreal-ue5-ci.ps1` fails with the PID when an editor from the same
  engine still holds the work directory, instead of dying inside `Remove-Item`.

### Added — UE 5.8 in the CI matrix

- `unreal-ue5` runs once per entry in the `UE5_ENGINE_ROOTS` repository
  variable; it now holds 5.7.3 and 5.8.3. Both pass with the same sources
  (BuildPlugin for editor and game targets, every shader compiled, both
  automation tests, the sample material) — no version gate was needed.

### Fixed — smooth union / intersection / subtraction: one operation order everywhere

- `smooth_min` / `smooth_max` computed `max(k - |a-b|, 0) / k` while the
  compiled, SIMD, BVH and JIT evaluators computed `max(1 - |a-b| * (1/k), 0)`
  from a precomputed reciprocal. The same law, a different f32 rounding: the
  tree and the bytecode disagreed by 1 ulp on five corpus nodes
  (`smooth_intersection`, `smooth_subtract`, `scale_smooth_union`,
  `scale_nested`, `nested`), which breaks the bit-exactness every evaluator
  has owed since 3.1.0. The tree now computes the same reciprocal and calls
  the same law; the SIMD hot path is unchanged. The scalar JIT codegen and
  the GLSL / WGSL / HLSL emit follow the same operation order.
- `tests/test_det_golden.rs` re-pins those five nodes (bits change on every
  platform; the law does not).
- New `tests/test_asdf_roundtrip_parity.rs`: save → load → compile → evaluate
  must be bit-identical to the in-memory tree for every corpus node. Nothing
  covered that path, which is exactly what the UE5 plugin runs.
- `AliceSDF.Unreal.FfiCorpusParity` found this; `test_det_parity`'s sample
  points never hit an affected point.

### Fixed — HLSL transpiler emitted `mix`

- `Morph` emitted GLSL's `mix`, which does not exist in HLSL — the generated
  shader did not compile in Unreal. `ShaderLang` gains `mix_expr` (HLSL:
  `lerp`), like the existing `modulo_expr` / `atan2_expr`.

### Added — UE5 plugin: icon, sample material, working install

- `Resources/Icon128.png`: the plugin browser icon, raymarched by this crate
  (`examples/unreal_plugin_icon.rs`, `--features image`) — a smooth union
  with a ring cut out of it.
- `Shaders/Public/AliceSdfSample.ush` + `Content/Python/create_alice_sdf_sample_material.py`:
  a Custom-node shader and the editor script that builds
  `/Game/AliceSDF/M_AliceSDF_Sample` around it (unlit, two-sided). CI runs
  the script headless and fails if the material does not compile.
- `Config/FilterPlugin.ini`: `ThirdParty/` and `Content/Python/` are kept when
  UAT packages the plugin (the filter drops `/ThirdParty` by default, so a
  packaged plugin had no native library and its module failed to load with
  `GetLastError=126`).
- `AliceSDF.Build.cs` delay-loads `alice_sdf.dll` and stages it next to the
  module; `FAliceSdfModule` looks in `Binaries/<platform>` then
  `ThirdParty/.../<platform>` and logs which one it loaded.
- `release.yml` packages the UE5 zip with `--features unreal`, ships
  `Shaders/`, `Config/` and — on Windows — the import library
  `alice_sdf.lib`, and fails if any of them is missing. Without those a user
  could not build the plugin at all.
- `docs/UNREAL_ENGINE.md`: install section rewritten to the real layout
  (the zip, `ThirdParty/`, delay load, the log line to check), Blueprint
  section replaced with the component's actual 124 nodes and palette groups.

### Fixed — Unreal plugin (it had never compiled on UE 5.7; the DLL in the repo was 1.7.2)

- Runtime (game) target: `SetActorLabel` (editor-only) behind a
  `WITH_EDITOR` helper; `UStaticMesh::SetNumSourceModels` /
  `CreateMeshDescription` / `CommitMeshDescription` / `Build` are editor-only,
  so `AAliceSdfNaniteActor` now builds a plain static mesh with
  `BuildFromMeshDescriptions` in cooked games (Nanite still needs the editor
  pipeline); `NaniteSettings` through the 5.7 accessors.
- `UAliceSdfComponent` derives from `USceneComponent` (it evaluated with
  `GetComponentTransform()`, which `UActorComponent` does not have).
- `FSceneViewExtensions::NewExtension`, `FRHIBufferCreateDesc::CreateVertex`
  2-argument overload, `SceneViewExtension.h` instead of the private
  `SceneRendering.h`, missing `Misc/Paths.h`, `ExportObj/Glb/Usda/Fbx`
  parameter `Bounds` shadowed `USceneComponent::Bounds` (now `HalfExtent`).
- Cargo feature `unreal` = `ffi` + `hlsl` + `glsl` + `gpu`: the plugin calls
  `alice_sdf_to_glsl` / `alice_sdf_to_wgsl` (`GenerateGlsl` / `GenerateWgsl`)
  which `ffi` + `hlsl` did not export (link error). This is what
  `scripts/build_ue5_plugin.sh` always built.
- `include/alice_sdf.h` declares the eight exported functions it was missing
  (`alice_sdf_save_abm` / `load_abm` / `export_unity[_binary]` /
  `export_ue5[_binary]` / `save_lod_chain` / `load_lod_chain`); the plugin
  copy of the header was 15 lines behind and is now identical.
- `AliceSDF.uplugin`: `VersionName` 1.7.2 → crate version, `EngineVersion`
  "6.0.0" → 5.7.0.

### Added — VRChat package: the four static samples (Basic / Cosmic / Fractal / Mix)

- Working collision in VRChat for the first time (the base `AliceSDF_Collider`
  gets a UdonSharp assembly definition, the samples override `Evaluate`, the
  body is sampled and the deepest wall pushes sideways), Cosmic / Mix colliders
  carry the full animated shader law (`animTime`), shaders get `Cull Off` /
  closest-approach acceptance / `_LightDir`, generator scenes are playable
  (descriptor, floor or viewing platform). Details in `vrchat-package/CHANGELOG.md`.
- `examples/vrchat_{basic,cosmic,fractal,mix}_golden.rs` and
  `vrchat-package/HostTests~/StaticParity`; `scripts/vrchat-host-parity.sh`
  runs all seven samples.

### Added — VRChat package: DeformableWall sample brought to the Mochi level

- The player collides with the dented wall (not the flat box), the body is
  sampled feet to eyes and pushed out sideways, walking in presses a body
  groove into the wall, desktop punching with the mouse, `[UdonSynced]`
  dents with strength (not `Time.time`) so they sync, no drilling through
  the thin wall, `Log Events`, shader `Cull Off` / closest-approach
  acceptance / hard-union AO / soft shadow. Details in
  `vrchat-package/CHANGELOG.md`.
- `examples/vrchat_deformable_wall_golden.rs` (5100 points, four dents) and
  `vrchat-package/HostTests~/DeformableWallParity` (29-check scenario);
  `scripts/vrchat-host-parity.sh` runs it.

### Added — VRChat package: TerrainSculpt sample brought to the Mochi level

- The player stands on the sculpted terrain: an invisible support box follows
  the player onto the SDF surface every frame (dig under yourself and you
  fall, build under yourself and you are lifted), desktop sculpting with the
  mouse, `[UdonSynced]` sculpt buffer, `Log Events`, shader `Cull Off` /
  closest-approach acceptance / hard-union AO / soft shadow. Details in
  `vrchat-package/CHANGELOG.md`.
- `examples/vrchat_terrain_sculpt_golden.rs` prints the terrain after six
  fixed sculpts from `alice_sdf::eval`; `vrchat-package/HostTests~/TerrainSculptParity`
  compiles the UdonSharp collider against the shared UnityEngine stub and
  checks its `EvaluateSdf` against that golden (4335 points, 1e-5) plus a
  sculpt / stand / wall / buried scenario. `scripts/vrchat-host-parity.sh`
  runs every sample project.

### Fixed — `Terrain` shader emit was invalid GLSL/WGSL/HLSL, and its law drifted from the CPU

The `Terrain` arm in `transpiler_common.rs` wrote raw GLSL (`float`, `vec2`,
`for(int`, an undefined `vnoise`) regardless of the target language; naga
rejected it in GLSL (unknown function) and WGSL (invalid syntax). The CPU
law separately hashed with `fract(sin(·)·43758)`, the same GPU/CPU-divergent
pattern `SurfaceRoughness` moved away from for `hash_noise_3d` (PCG lattice
hash) before this.

- `SdfNode::Terrain`'s 3-octave fbm now samples `hash_noise_3d` on the xz
  plane (y held at 0, degenerating the trilinear blend to bilinear) on the
  CPU (`eval/mod.rs`) **and** through `ensure_helper("hash_noise")` +
  portable `ShaderLang` ops in the shader emit — same law, one source, no
  language-specific string left in `transpiler_common.rs`.
- `TERRAIN_FBM_GRAD` (the `eval_lipschitz` bound) updated 3.208 → 6.42: the
  new noise's range is `[-1, 1]` (was `[0, 1]`), doubling the per-axis
  gradient bound; `lipschitz_claim_bounds_every_difference_quotient`
  verifies it empirically.
- `Terrain` added to the shared GPU-oracle corpus
  (`tests/common/corpus.rs`) — it was the one node in the "144 nodes
  transpile, corpus oracle covers them" (3.0.0) claim the corpus never
  actually exercised. `every_corpus_node_matches_cpu_through_wgsl` /
  `_through_glsl` now cover it; `test_det_golden.rs` gained its pin.
  `test_det_parity.rs` and `test_evaluator_opcode_parity.rs` (which iterate
  the same corpus but exercise the bytecode compiler) now skip nodes with
  no bytecode law — `CompileError::UnsupportedPrimitive` — instead of
  assuming every corpus entry compiles.

### Fixed — `Elongate` shader law disagreed with every other evaluator

The tree evaluator, VM bytecode (`real::elongate`) and both Cranelift JITs
all implement IQ's cheap elongate — `q = p - clamp(p, -a, a)`, the child
evaluated as-is at `q` — but the GLSL/WGSL/HLSL transpilers emitted IQ's
*exact* elongate (`q = max(abs(p) - a, 0)` plus a `min(max(q), 0)` box
correction on the returned distance), a different law that disagreed by up
to 1.0 on interior points (`elongate(1,2,3, sphere(1))` at
`(0.011, 0.666, 0.515)`: cpu `-1.0`, shader `-1.99`). The shader emit now
matches the other four evaluators; `every_grammar_construct_matches_cpu_on_gpu`
(alice-lol) and the corpus GPU-parity tests confirm it.


### Fixed — `SdfField::distance_and_normal` が `distance` と違う距離を返していた

`CompiledSdfField` は trait の既定実装を `eval_compiled_distance_and_normal` で
上書きしていた。この helper は法線の四面体標本 4 点を再利用し、その**平均**を中心の
距離とする (helper 自身の doc が「approximated」「error ≈ O(epsilon²)」と明記している)。
評価回数は 5 → 4 に減るが、返る距離は `distance` と別物になる。

trait の契約は既定実装が定めている (`(self.distance(..), self.normal(..))`、doc は
override の目的を「both を効率よく計算できる実装のため」と書いており、別の答えを
返すためではない)。呼び出し側はそれを前提にしていた —
`alice_physics::sdf_adaptive::AdaptiveSdfEvaluator::evaluate_and_cache` (1.4.0) は
`distance_and_normal` の距離を cache に格納し、`collide_point_sdf` は `distance` から
接触深さを導く。同じ点について 2 経路が食い違っていた。

実測 (sphere ∪ box、24 点): 上書き側の距離は全点で `distance` より大きく、差は
4.768e-7 〜 7.451e-7。全点が同符号なので丸めではなく偏り。

#### 対応

上書きを削除した。trait の既定実装が適用され、距離は `distance` と同じ 1 回の評価に
なる (法線は従来と同じ四面体 4 点)。評価回数は 4 → 5 で、これは厳密な距離の値段。
2 経路が構造的に一致するので、あとから片方だけがずれる余地がなくなる。

`eval_compiled_distance_and_normal` 自体は残している。4 評価の近似であることが doc に
書かれており、そのトレードオフが欲しい呼び出し側は名前で指定できる。

### Added — SDF ↔ Physics 境界の決定論オラクルと、extern 宣言の順序ガード

`tests/test_physics_bridge_determinism.rs` (9 本)。両クレートが別々に bit-exactness を
主張しているが、その合成は誰も測っていなかった (`grep to_bits|deterministic|bit` が
bridge の 2 file で 0 件)。境界は
`Vec3Fix → world_to_local → f32 → SDF 評価 → f32 → Fix128` で、この f32 区間が
Fix128 エンジンの中の非 Fix128 層にあたる。

反復 64 回 / 訪問順の逆転 / collider 再構築 / 衝突 32 回で `Contact` の全 10 word が
bit 一致すること、bridge が `eval_compiled` と bit 一致すること、球の `|p| − r` が
**bit 厳密**に一致すること (既存 bridge test の許容 0.01 を置き換え)、両 stencil が
解析法線に一致して**外向き**であること、表面上の点が衝突しないこと (内側の陽性対照
付き) を確認する。破壊試験 3 種で red を実測済 — うち法線の符号反転は 9 本中 1 本しか
落ちないので、符号は単位長でも反復でも閉形式でも見えず、解析方向との内積だけが
捕まえる。

`scripts/abi_decl_check.py`。`tests/` と `examples/` の `extern "C"` 宣言を
`src/ffi/` の export と**順序付きの型列**で突き合わせる。`scripts/unreal-abi-check.sh`
step 2a は header と C# を見るので、test 内の宣言は第 3 の宣言箇所として無検査だった。
引数順の誤りは SysV (macOS / Linux) では整数と浮動小数でレジスタバンクが分かれ
独立に採番されるため値が正しいレジスタに着いて通り、Microsoft x64 は位置でスロットを
決めるので落ちる。型の集合では検出できない。preflight の source guard 群と ci.yml に
配線した。

### Fixed — test の `alice_sdf_repeat_finite` 宣言が引数順を入れ替えていた

`tests/test_binding_oracle.rs` の宣言が `(node, f32 × 3, u32 × 3)` で、export は
`(node, u32 × 3, f32 × 3)`。run 36680136653 で Windows のみ red になり、
`repeat_finite` が `Vec3(1,0,0)` で FFI -3.0000001e-1 / native 4.0000004e-1。
実装とヘッダと C# バインディングはいずれも正しく、test 側の宣言と呼び出しのみ修正。

### Fixed — FFI / 公開 Rust API から到達する SoA の境界外 read+write

`compiled::eval_compiled_batch_soa_raw` が `count` を 8 の倍数へ切り上げて走るので、
`count % 8 != 0` のとき caller の allocation の外へ最大 7 要素を読み書きしていた。

関数自身の doc は `out_ptr` must point to valid memory of at least `count` f32s と
書いており、`count should be a multiple of 8` は **should** (助言) だったので、
doc どおりに `count` 長の buffer を渡した caller が境界外に落ちる。到達経路は 3 つ:

- `compiled::eval_compiled_batch_soa_raw` の直接呼び出し (`pub`、prelude 経由で再公開)
- `alice_sdf_eval_soa` (C ABI、`count < 1024` の逐次分岐)
- `alice_sdf_eval_animated_batch_soa` (C ABI、恒等変換の分岐 — こちらは 1024 の閾値が
  無いので `count = 1025` でも起きる)

`count` ぴったりの `Vec<f32>` で呼ぶと SIGSEGV になる。範囲内 (`0..count`) の値は
修正前から native `eval_compiled` と bit 一致しており、壊れていたのは範囲外だけ。

⚠️ この形は **publish 済の 3.1.0 と source が同一**だった。3.1 系にも同じ修正を
backport して **3.1.1 として publish 済** (2026-09-30、tag `v3.1.1`、branch
`3.1.x`)。3.1.0 を使っている呼び出しは 3.1.1 に上げるだけで直る。

#### 対応

- raw 関数を `count & !7` の SIMD + `count % 8` の scalar tail に変更。8 の倍数での
  出力は修正前後で bit 一致 (`count=8` digest `5ce40a654075eaa6`、8/64/256/1024/4096)。
- 矛盾していた記述を整理 (`count should be a multiple of 8` / loop 内の SAFETY の
  「rounded up to multiple of 8」/ FFI 側の「handles the 8-wide alignment internally」)
  と、計算して捨てられていた `_aligned_count` / `_simd_count` を削除。
- `tests/test_binding_oracle.rs` に `soa_must_not_write_past_the_documented_count` と
  `exact_size_buffers_survive_and_agree` を追加。修正を戻すと macOS の malloc が
  heap 破壊を検出して SIGTRAP になる。

### Fixed — terrain / destruction の 3 件

- `destruction::remesh_chunk` の marching cubes cell 原点が voxel の角だったのに、
  corner に入れる距離は voxel 中心で評価したものだったため mesh が半 voxel 斜めに
  ずれていた (球 r=1.5、res16 で `max|‖v‖−r|` 0.218 → 0.0040)。
- `destruction::fracture` が voxel size の x 成分を 3 軸すべてに流用していたため、
  非等方 voxel で破片表面の位置と大きさが狂っていた (grid `[24,48,24]` で
  軸平行表面積 71.11 → 42.67、Cauchy の射影公式 `6πr²` = 42.41)。
- `make_tangent_basis` が左手系を返していた (`v = u × n` なので `u × v = −n`) ため
  破片 mesh の面が内向きに巻かれていた (符号付き体積 −14.41 → +14.17、解析 +14.14)。
  既存 unit test は直交性だけを見て利き手を見ていなかった。

### Fixed — gi / volume の 3 件

- `ConeTraceConfig::num_cones` は pub field なので 0 が公開 API から到達するが、
  `generate_cosine_cones` が `num_cones - 1` を u32 で引いていた (release では
  0..4_294_967_295 の loop)。
- `IrradianceGrid::sample` が格子 index を `clamp(0, n-1)` する一方、三線形補間の
  重みを clamp **前**の座標から作っていたため、probe 中心面を境に 1 セル分の
  不連続が出ていた (2³ / [4,3,2] grid では probe 中心が f32 で厳密に境界を踏むので
  現れず、step 0.8 の 5³ で顕在化する)。
- `volume::generate_mip_chain` / `_distgrad` が常に 2×2×2 の子を読むので、level の
  解像度が奇数になる遷移で最後の slab が落ち、min-downsample が footprint の下界に
  ならなくなっていた (`[10,10,10]` も 10→5 で該当)。module doc の
  "preserves the SDF distance property (closest surface wins)" に反する。
  解像度 chain と段数は不変で、2 冪 (非立方 `64×64×1` 含む) の出力も不変。

### Changed — SH1 の球面調和射影に立体角測度を入れた

`SH1::project` が `∫ f Yᵢ dΩ` の測度を落としていたため、再構成が `f(e)/(4π)` に
なっていた (間接光が約 0.08 倍)。射影側に `4π` を入れ、再構成 (`evaluate`) は
標準の `Σ cᵢ Yᵢ` のまま据え置いた。`sample` / `bake_irradiance_grid` の値は
ちょうど 4π 倍になる (110 値で実測、最悪相対偏差 4.5e-6、符号と方向は不変)。

### Added — 解析解 oracle 4 file (合計 68 本) と、それを走らせる CI step

`tests/test_terrain_destruction_oracle.rs` (28) / `tests/test_gi_volume_oracle.rs` (23) /
`tests/test_binding_oracle.rs` (9 + godot in-module 4) /
`tests/test_hlsl_blinkscript_parity.rs` (8) + `tests/common/hlsl_cpu_shim.h`。

- HLSL / BlinkScript は naga frontend が無いので、emit した HLSL を clang++ で
  native 実行して CPU 法と突き合わせる (GPU / dxc / Vulkan 不要)。corpus 152 node を
  `Hardcoded` / `Dynamic` 両モードで通し、glslang + spirv-val が 304 shader を受理。
  `lattice_deform` のみ 1.276e-4 の乖離が残る (中心差分の f32 増幅、Metal も同じ node を
  最悪値に挙げる) ので実測値付きで許容を開示し、他 151 node は 1e-4 → 1e-5 に締めた。
- 既存 unit test の許容を 6 箇所締めた (`test_bilinear_sample` / `test_bicubic_sample` は
  ±1.0 → 1e-5、`test_normal_at` / clipmap は 0.1 → 1e-6 等)。`test_remesh_chunk` は
  assert が 0 個だったので球面上判定を入れた。
- `test_erosion_smooths_terrain` の「Erosion should generally reduce roughness」は
  既定では成り立たない (hydraulic 侵食は roughness を 1.5〜2.7 倍にする)。平滑化するのは
  thermal 単独なのでそちらに分離した。assert が `>= 0.0` だったため露呈していなかった。
- `ci.yml` に oracle step 5 本を追加し、clippy の feature 集合に `blinkscript` を
  加えた (`src/compiled/blinkscript` は一度も lint されておらず、error 3 件が
  残っていた)。`gpu-parity` job に volume の GPU bake oracle を追加。

### Changed — preflight の --quick が最低限の test を走らせる

`scripts/preflight.sh --quick` は test 群の手前で exit していたため、pre-push hook が
`cargo test` を 1 本も実行していなかった。`cargo test --lib` を `--quick` 内に戻し、
成功時の出力に走った step と走っていない step を列挙するようにした。

### Changed — CI が JIT の parity oracle と FFI の unit test を実行するようになった

oracle の棚卸しで、**書かれているが一度も実行されていない oracle** が 2 箇所見つかった。

**(1) JIT backend の parity arm**。`tests/test_det_parity.rs` と
`tests/test_evaluator_opcode_parity.rs` は `JitCompiledSdf` (scalar) と
`JitSimdSdf` を tree evaluator と突き合わせる arm を持つが、いずれも
`#[cfg(feature = "jit")]` で囲まれている。CI の `cargo test --tests` は
default feature で走り、`jit` 付きの `cargo test` は 1 つも無かったため
(`Build (JIT) [T3]` は `cargo build`)、7,675 行の JIT 経路の値の正しさは
検証されていなかった。

test 本数では退行を検出できない点に注意。default でも `test_det_parity` は
1 件、`test_evaluator_opcode_parity` は 10 件を報告し、消えるのは中の比較 arm
だけである。`test_relaxed_tracing` だけが 8 → 9 と変化する
(`relaxed_tracing_matches_oracle_jit`)。

**(2) `src/ffi` の unit test**。15 本あるが `ffi` feature 付きの `cargo test` が
CI に無く (`Build (FFI + shaders)` は `cargo build`)、registry の handle
lifecycle・panic sentinel・poisoned lock 耐性・compiled / batch / SoA eval が
一度も実行されていなかった。

#### 対応

- `ci.yml` に `Test (integration, JIT parity oracles) [T3]` を追加
  (`cargo test --features jit --test test_det_parity --test
  test_evaluator_opcode_parity --test test_relaxed_tracing`)。
- `ci.yml` の `Build (FFI + shaders)` を `Test (FFI + shaders)`
  (`cargo test --lib --features "ffi,hlsl,glsl"`) に格上げ。
- `scripts/preflight.sh` に同じ 2 step を追加し、ffi を builds 群から test 群へ移動
  (ci.yml と逐語対応を維持)。

crate の主張に対して落ちる job を持たせる原則の適用であり、公開 API の変更は無い。

### Fixed — mesh → SDF の符号が三角形の巻き順に依存し、三角形距離の edge clamp も誤っていた

外部で生成された mesh を SDF として取り込む経路が 2 つの独立した理由で壊れていた。

**(1) 符号が巻き順依存**。`MeshBvh::signed_distance` は `|d|` 最小の三角形を選び、
その面法線との内積で符号を決めていた。面法線は頂点の並び順で向きが変わるので、
場が幾何ではなく入力の書き方の関数になっていた。結果として

- 開曲面 (囲む体積を持たない板) の片側に架空の内部ができる、
- index を逆順にすると場全体が反転する、
- 独立に巻かれた部品を連結した mesh では、巻き方が食い違う部品だけ内外が裏返る。

**(2) `Triangle::signed_distance` の edge 2-0 の clamp が符号反転後に適用されていた**。

```rust
let t2 = clamp01((-v02).dot(p2) / v02.length_squared().max(1e-10));
let d2 = (p2 + v02 * t2).length_squared();
```

真のパラメータ `t*` に対して `clamp01(-t*)` を取ってから符号を戻すので、
`t* ∈ [0,1]` (投影が辺の内側) では `v2` までの距離に落ちて**過大**、`t* < 0`
では無限直線上の点までの距離になり**過小**になる。この結果、閉じていて巻き順も
正しい箱ですら符号が壊れていた ─ 半径 (0.7, 0.5, 0.9) の箱を格子 3165 点で
解析解と突き合わせると **207 点 (6.5%) の符号が食い違い**、最大誤差 2.299。

#### 対応

- `Triangle::closest_point` を Voronoi 領域形 (Ericson, *Real-Time Collision
  Detection* §5.1.5) で追加し、`signed_distance` / `unsigned_distance` の大きさを
  そこから導くようにした。`MeshBvh::closest_point` / `MeshBvh::unsigned_distance`
  も符号ロジックを経由しない専用探索にした。
- `src/mesh/mesh_sign.rs` を追加。`SDF(x) = (1 − 2·T(x)) · UDF(x)` で、`T` は
  padding 付き bounding box の外周から 6 近傍 flood fill して得る到達可能性。
  面法線を一切読まないので巻き順・閉曲面かどうか・連結成分数のいずれにも依存
  しない。表面帯のセルは最近接点から離れる向きへ 1 セルずつ歩いて分類済セルの
  label を取るので、セル未満の分解能で符号が決まる。
- **(breaking)** `MeshToSdfConfig` に `sign_mode` / `sign_flood_fill_resolution`
  を追加。`MeshToSdfConfig::default()` は従来どおり
  `MeshSignMode::NearestFaceNormal` なので、既定の呼び出しの符号規則は変わらない
  (符号の大きさは (2) の修正ぶん正しくなる)。⚠️ **ただし struct に
  `#[non_exhaustive]` が無いので、全 field を並べる struct literal
  (`MeshToSdfConfig { a, b, c }`) は field 不足で compile error になる**
  (`cargo semver-checks` の `constructible_struct_adds_field` 2 件)。
  `..Default::default()` / `..MeshToSdfConfig::fast()` を付けるか、
  constructor (`fast()` / `accurate()` / `hybrid()` / `topology_robust()`) を使う。
  依存 30 repo を走査した範囲では網羅 literal は **0 件** (唯一の下流
  `ALICE-Manga/src/vrm_import.rs` は constructor 経由)。
- **(breaking)** `MeshToSdfConfig::accurate()` の符号規則を
  `ExteriorFloodFill` に変更。「accurate」が名乗るとおり、巻き順ではなく幾何から
  符号を決めるようになった。⚠️ **`accurate()` を呼んでいる既存コードは戻り値が
  変わる** (実 caller: `ALICE-Manga/src/vrm_import.rs:1167`)。3.x の挙動が要る
  呼び出しは `cfg.sign_mode = MeshSignMode::NearestFaceNormal` で戻せる。
  ⚠️ 4.0.0 公開前の版 (3.1.1 まで) の利用者には major bump なので自動では
  届かないが、**3.1.x から 4.0.0 へ上げた時点で届く**ので breaking として扱う。
  副作用として、`accurate()` は退化 mesh (bounding box の全辺が 0 / 非有限) で
  `MeshSdf::new` が `None` を返すようになった。理由が要る場合は
  `MeshSdf::try_new` を使う。
- `MeshToSdfConfig::topology_robust()` を追加。`accurate()` への委譲で、
  「精度」ではなく「mesh の出所が信用できない」ことが選択理由である呼び出し側の
  ための明示的な別名。
- `MeshSdf::try_new` を追加。flood fill の構築失敗を従来規則へ黙って落とさず
  `MeshInputError` で返す。

分解能が精度の唯一のつまみで、およそ 2 セルより薄い壁や隙間は潰れる。
`sign_flood_fill_resolution` (既定 64) を最も薄い形状が数セルにまたがるまで上げる。

#### 検証

`tests/test_mesh_sign_topology.rs` は期待値を解析解 (厳密な箱 SDF / 矩形までの
厳密距離) だけから作り、実装を呼んで期待値を組み立てていない。

| oracle | 実装前 | 実装後 |
|---|---|---|
| 開曲面に内部が無い | 符号不一致 1014/2116 (47.9%) | 0 (0.0%)、最大誤差 0.000e0 |
| 巻き反転で場が不変 | 2135/2197 点が相違 (最大 2.691) | 0/2197、bit 一致 |
| 分離部品 + 巻き不一致 | 符号不一致 1797/3325 (54.0%) | 0 (0.0%)、最大誤差 0.000e0 |
| 閉じた箱 = 解析解 | 符号不一致 207/3165 (6.5%) | 0 (0.0%)、最大誤差 0.000e0 |
| 厚さ 0.1 板の内部が負 | (通過) | 通過 |

決定論は保たれる。セル占有は独立な per-cell 距離クエリ (リダクション無し)、
flood fill は整数 index のみ、浮動小数は加減乗除・比較・`sqrt` だけを使う。

### Fixed — `alice_sdf_mirror` / `alice_sdf_repeat_finite` の宣言が Rust の署名と食い違っていた

`include/alice_sdf.h` と `bindings/AliceSdf.cs` が `alice_sdf_mirror` を
`float mx, float my, float mz` と宣言していたが、Rust の export は
`fn(SdfHandle, u8, u8, u8)` である。整数引数と浮動小数引数は AArch64
(x0-x7 / v0-v7) でも x86-64 SysV (rdi-r9 / xmm0-7) でも別のレジスタ群で渡る
ので、`float` として呼ぶと Rust 側は整数レジスタの残留値を mirror フラグとして
読む。crash しないため気付きにくい。

arm64 / release で実測した値 (半径 1 の球を (2,2,0) へ移動 → X 軸だけ mirror
指定 → 点 (-2,-2,0) を評価、3 通りが分離する):

| 呼び方 | 距離 | 実際に立った軸 |
|---|---|---|
| mirror なし | 4.656854 | — |
| `uint8_t` 宣言 (修正後) | 3.000000 | X のみ = 指定どおり |
| `float` 宣言 (修正前) | -1.000000 | X と Y |

残留レジスタの値は呼び出し文脈で決まるので結果は文脈依存だが、**「毎回変わる」
わけではない**。上記の測定では 3 回連続で同じ -1.000000 が出た。つまり
「たまたま動いているように見えて、呼び出し位置を変えると軸が変わる」種類の
壊れ方であり、テストが 1 箇所でも通ってしまえば見逃される。

影響していた経路は 2 つで、どちらも宣言を直すだけで閉じる:

- Unity — `bindings/README.md` が `bindings/AliceSdf.cs` を Unity へコピーさせて
  いるので、`AliceSdf.Mirror()` がそのまま不定動作だった
- UE5 — `AliceSdfComponent.cpp` の呼び出しは `1u` / `0u` と整数で書かれていたが、
  可視のプロトタイプが `float` なので C++ が暗黙に `1.0f` へ変換していた
  (呼び出し側は変更不要)

`alice_sdf_repeat_finite` はヘッダが `int32_t`、Rust が `u32` で幅は同じだが
符号が違うので `uint32_t` に揃えた。C# 側は負の count が約 40 億回の反復に
回り込む代わりに `ArgumentOutOfRangeException` で落ちるようにしてある。

### Added — `unreal-abi-check.sh` に引数型の突合 (step 2a)

この食い違いが 1.7.2 から 4.0.0 まで残った理由は gate の側にある。
`scripts/unreal-abi-check.sh` の step 2 はヘッダの宣言と cdylib の export を
**名前集合でのみ**照合しているので、型が何であっても通る。step 1 の byte 一致も
複製ヘッダ同士の比較なので、両方が同じ間違いを持っていると検出できない。

step 2a を追加し、`include/alice_sdf.h` の各宣言の**引数型**を `src/ffi/**` の
`extern "C"` 署名と突合するようにした (現在 175 関数で不一致 0)。C の `float *`
と Rust の `*const f32` のような表記差は正規化し、`float m[16]` のような配列
記法はポインタと同一視する。名前集合の照合は引き続き step 2 の担当 (実際の
cdylib を見るのでソース解析より強く、feature gate された export も扱える)。

追加後にわざと壊して red を実測した: mirror を `float` に戻す / 引数の数を
減らす / ポインタを値にする / 符号だけ変える (`uint32_t` → `int32_t`) の 4 つは
いずれも fail し、配列記法への書き換えは green のまま (誤検出なし)。

## [v3.1.0] - 2026-09-17

### Changed — cross-platform bit-exact evaluation (alice-det-math)

Distances move in the last ulp everywhere a law calls a transcendental
(twist / bend / TPMS / polar / helix / smooth-exp / …) and wherever
`mul_add` was used; no law changed its meaning. Goldens re-pinned.

- Every transcendental in the evaluator and law directories (`primitives` /
  `modifiers` / `operations` / `eval` / `compiled` / `raycast` / `sdf2d`) goes
  through [`alice-det-math`](https://crates.io/crates/alice-det-math) 0.2
  (scalar and `f32x8`, the crate `alice-physics` 1.4 uses), and `a * b + c`
  is two roundings everywhere (`mul_add` removed — it fuses on FMA hardware
  and not elsewhere). `Real` for `f32` / `f32x8` (`sin_cos` / `atan2` / `exp`
  / `ln` / `round` / `mul_add`) dispatches to it; the SIMD table's own `wide`
  sin / cos are gone.
- The tree evaluator, compiled scalar, `f32x8` SIMD, BVH and Cranelift SIMD
  JIT are **bit-identical** to each other (`tests/test_det_parity.rs`, 144
  corpus nodes × 266 points, `to_bits()` equality) and across x86_64 /
  aarch64 / wasm32 (`tests/test_det_golden.rs`, one SHA-256 per corpus node
  on a libm-free grid; verified on aarch64 and x86_64 via Rosetta locally,
  CI lanes pin it). `Scale` in the tree evaluator uses `p * (1/s)` like the
  compiled evaluators (was `p / s`).
- SIMD JIT: `simd_sincos_approx` (unreduced Taylor, 4e-6 abs error, and its
  `bitselect` masks were never bitcast — twist / bend did not compile on
  Cranelift 0.113, which the tolerance test skipped silently) is replaced by
  the det_math law emitted as IR; `fma` → `fmul` + `fadd`; the length,
  smooth-blend, rotate, cone, rounded-cone and pyramid arms follow the law's
  operation order (`(x*x + y*y) + z*z`, `h = max(1 - |a-b|·rk, 0)`,
  quaternion rotation, division by `k2·k2` / `m2`). Ellipsoid has no JIT arm
  any more: the scalar law is the exact Eberly distance, the JIT arm was the
  IQ approximation (up to 30% off). The scalar tree JIT (`JitCompiledSdf`)
  is held to 1e-5 only (3.2.0).
- Shaders: `alice_atan2` helper (WGSL / GLSL / HLSL) returns the CPU law's
  exact constants on the axes (`atan2(±0, x<0) = ±π`, `atan2(y, ±0) = ±π/2`),
  so polar / polygon / helix sector ties on an axis snap like the CPU
  (`polar_axis_ties_gpu_match_cpu`, WGSL and GLSL). Hardcoded constants are
  printed with round-trip precision (`6.2831855`, was `{:.6}` → `6.283185`,
  which alone flipped a sector at π).
- `security-audit.yml` / preflight: `scripts/det_math_guard.py` fails on any
  libm method call or `mul_add` in the bit-exact directories.
- `benches/sdf_eval.rs`: `transcendental_laws` group (6 laws × scalar / SIMD /
  JIT). SIMD is faster than 3.0.0 (gyroid 121 → 20 µs / 4096 pts, twist 22 →
  12, exp-smooth 23 → 24); scalar is 1.3–2.4× slower (libm 1.5 ns vs det_math
  3.6 ns per `sin`) — the price of the guarantee.

### Fixed

- `tests/test_relaxed_tracing.rs`: the judge re-evaluated a hit with the
  un-renormalised direction (the marcher normalises), which crossed the 1e-4
  band on a grazing gyroid ray by 2e-9.

### Fixed — VRChat package: Mochi sample pushed the player off the floor every frame

- `SampleMochi_Collider.PostLateUpdate` tested the player's feet (5 cm under
  the player position) against `EvaluateSdf`, which includes the ground
  plane `y = 0`. Standing on the world floor was therefore a permanent
  penetration: the collider teleported the player up, gravity brought them
  back, and the view bobbed for as long as they stood still — in every
  version of the sample. The player now collides with the mochis only
  (`EvaluateMochiSdf`, the same smooth union without the ground); the floor
  is the world's own collider at the same height, and `EvaluateSdf` is
  unchanged for the shader / golden parity. `HostTests~/MochiParity` checks a
  floor point away from the mochis is not a penetration. Found in the VRChat
  client after the first Build & Test.

### Fixed — VRChat package: Runtime assembly could not compile inside a VRChat project

- `vrchat-package/Runtime/AliceSDF.Runtime.asmdef` defines `UDONSHARP` when
  `com.vrchat.worlds` is present but referenced no assemblies, so
  `using UdonSharp; using VRC.SDKBase; using VRC.Udon;` in
  `Runtime/Udon/AliceSDF_Collider.cs` did not resolve and the package failed
  to compile in every VRChat world project. `references` now lists
  `UdonSharp.Runtime`, `VRC.Udon`, `VRC.SDKBase`. The host-side parity
  harness could not see this because it compiles without `UDONSHARP`.
  Verified in Unity 2022.3.22f1 + VRChat SDK 3.10.1 (`AliceSDF.Runtime.dll`
  builds, Mochi sample renders in ClientSim, `PostLateUpdate` pushes the
  uniforms every frame).

### Added — VRChat package: host-side parity of the Mochi collider

- `examples/vrchat_mochi_golden.rs` prints the Mochi scene from
  `alice_sdf::eval`; `vrchat-package/HostTests~/MochiParity` compiles the
  UdonSharp collider against a UnityEngine stub and checks its
  `EvaluateSdf` against that golden (1521 points, 1e-5) plus a
  grab / split / merge scenario. `scripts/vrchat-host-parity.sh`, CI job
  `vrchat-host` (setup-dotnet), preflight step.

### Changed — VRChat package: Mochi sample

- Same features, tighter code: ground / mochi shading blends by the
  smooth-union factor (no seam at the neck), LOD-scaled normal epsilon,
  value-noise ground, soft contact shadow, light / fog as material
  properties; the collider owns `blendK` / `groundK` and pushes them to
  the material, hand state is indexed by hand, settle is frame-rate
  independent. Host-verified (glslang HLSL, .NET compile, `EvaluateSdf`
  vs `alice_sdf::eval` parity 6e-8 on 1521 points); not compiled in
  Unity here (see `vrchat-package/CHANGELOG.md`).

## [v3.0.0] - 2026-09-16

Every node kind is transpiled, `ShaderLang` is sealed (the reason for the
major bump), IFS / skinning made sound on the CPU.

### Changed — breaking: `ShaderLang` is sealed

- `compiled::transpiler_common::ShaderLang` is now sealed, like `Real`
  since 2.0: only `WgslLang`, `GlslLang` and `HlslLang` implement it. It
  was left unsealed in 2.0 by oversight; it gains a required method every
  time a node kind needs new syntax (four in this release), and an
  implementation has to supply every helper the emitted laws reference,
  so external implementations were never workable. cargo-semver-checks
  flags both the new required methods and the sealing as major, hence
  3.0.0 rather than 2.2.0. No dependent on crates.io or in the ALICE
  repositories implements the trait; consumers of `to_wgsl` /
  `to_glsl` / `to_hlsl` / `GpuEvaluator` are unaffected.

### Added — every node kind is transpiled (IFS, skinning, lattice, heightmap)

- The four kinds the shaders used to pass through unchanged are now emitted
  in WGSL / GLSL / HLSL: `IFS` (transforms and iterations unrolled as
  literals, `ifs_fold_with_scale` law, child ÷ accumulated scale),
  `SdfSkinning` (each bone's two column-major transforms, weighted mean),
  `LatticeDeform` (control points as a module-scope array, the FFD as a
  module-scope function called three times for the central-difference
  correction), `HeightmapDisplacement` (the map as a module-scope array,
  dominant-axis projection, bilinear sample). `shader_unsupported_nodes`
  now returns an empty list; the corpus oracles cover all 144 nodes.
- `ShaderLang` gained `cast_int` / `decl_int` / `global_float_array` /
  `global_vec3_fn`; `GenericTranspiler::globals` collects module-scope
  declarations that each language emits before `sdf_eval`.

### Fixed — IFS / skinning on the CPU (found by non-identity corpus entries)

- The corpus IFS and skinning entries used identity matrices, so nothing
  had checked the laws with real transforms. With a scale / rotate IFS and
  a two-bone skin: the scalar / SIMD / BVH VM dropped the IFS scale
  correction the tree applies (`/ max(scale, 1e-6)`, now applied at
  `PopTransform`); the interval evaluator treated both as identity maps
  (skinning is one affine map `A p + b` — pushed through exactly; IFS
  uses the hull of the box and its images, divided by the scale range);
  `eval_lipschitz` claimed `2 · L(child)` for IFS (unsound by 280× — the
  nearest-image choice jumps, so it is `INFINITY` like domain repetition,
  pinned set 14 → 16) and `L(child)` for skinning (now `L(child) · ‖A‖`).

## [v2.1.0] - 2026-09-16

Corpus-wide GPU execution oracle (WGSL + GLSL) and the 15 shader laws it
caught, rounded cylinder fixed on the CPU, step budget scaled with the
Lipschitz bound (the gyroid "8.3 % miss"), a slimmer scalar-VM frame, one
VRChat shader source.

### Changed — VRChat package: one shader source

- `vrchat-package/Runtime/Shaders/` is the only copy; the legacy
  `Assets/AliceSDF/Shaders/` fork (which alone had the PBR surface and the
  material-id ops) is merged into it and deleted, together with the June
  `.unitypackage` snapshot. Merged by hand, not yet compiled in Unity
  (see `vrchat-package/CHANGELOG.md`).

### Fixed — 15 shader laws that differed from the CPU (found by the new corpus GPU oracle)

- `tests/test_gpu_law_parity.rs` now runs **every corpus node** through the
  WGSL path *and* through the GLSL path (`GpuEvaluator::from_glsl_compute`,
  wgpu's `glsl` feature / naga glsl-in), on 1024 random points each. The
  hand-picked WGSL tests had covered the laws touched by specific fixes;
  the blanket run found 16 of 142 nodes drifting on WGSL (up to 3.4) and
  18 on GLSL. Fixed to the CPU law in all three transpilers: heart (the
  shaders had a different implicit-cubic heart), pie, vesica (axis and
  sign), box frame (typos in the `max` operands), lidinoid / IWP / FRD
  (different surfaces), columns union / intersection / subtraction (one
  helper per language, `hg_sdf` law; the GLSL helper also used `half`,
  a reserved word), bend (rotation sign), extrude (2-D child in XY, slab
  on Z), displacement (frequency 5, not 10), octant mirror (abs *and*
  sort), icosahedral symmetry (was a pass-through; now the fold).
- **Rounded cylinder was wrong on the CPU**: `sdf_rounded_cylinder` and
  the SIMD table carried IQ's `− 2·ra` literally, so `radius = 0.4`
  rendered as 0.8 on the CPU while the shaders (and the docs) meant 0.4.
  CPU and BVH AABB now use `ρ − radius + round_radius`.
- `compiled::shader_unsupported_nodes(&node)` / `SHADER_UNSUPPORTED`: the
  four node kinds the transpilers pass through unchanged (LatticeDeform,
  HeightmapDisplacement, SdfSkinning, IFS — per-node data with no shader
  binding); the emitted shader carries a comment, the oracles skip them.
- Exact ties (a sector boundary at `atan2 = π`, a columns cell boundary)
  are platform-dependent on the GPU: the corpus oracles use random points,
  `repeat_laws_gpu_match_cpu_at_ties` keeps pinning the WGSL tie behaviour.

### Changed — scalar VM transform frame slimmed

- `eval_compiled` (scalar) pushed opcode + 4 params + aux window on every
  transform; the frame is now the point plus the pushing instruction's
  index, read back at `PopTransform`. Against the tree walker on CSG
  scenes (release, single point): 5 primitives 1.02×, 10 → 1.19×
  (was 1.35×), 20 → 1.17×, 40 → 0.91× (was 1.05×). The 9/16 self-review's
  "still 30 % slower" is this push / pop pair per transform; the scalar VM
  is the front of the SIMD batch path (`eval_compiled_batch_simd`, ~4×
  the tree), which is where compiled evaluation pays.

### Changed — step budget scales with the Lipschitz bound

- `RaymarchConfig::with_bound` multiplies `max_steps` by the bound (steps
  are `d / L`, so the same count covers `1 / L` of the distance) and the
  default budget is 256 (was 128); `high_quality` 512; `relaxed` uses the
  same scaled budget. The 9/16 self-review's "gyroid: 8.3 % of rays lost,
  unchanged in 2.0.0" was budget exhaustion, not a law problem: on 3000
  random rays through `gyroid(1.0, 0.1)` the 1.x default (128 steps at
  L = √3) lost 0.7 %, 444 steps lose 0, and the only case left
  (`gyroid(2.0, 0.1)`: 2 / 2540) is a ray grazing the shell and creeping by
  ≈ ε per step, which 4096 steps resolve. Pinned by
  `tpms_default_budget_random_rays`. A ray that stops inside the ε band
  while grazing (`f = 7e-6`) is a hit by definition even when the first
  sign change is further along.

## [v2.0.0] - 2026-09-16

Taper is a distance bound (with a phantom-free singular plane), the four
breaking changes deferred through 1.x (sealed `Real`, private `CompiledSdf`
fields, `dep:` features, no `lazy_static` feature), dual contouring without
fins, one noise law for texture fitting with GPU parity.

### Changed — **breaking**

- `SdfNode::Taper` gained `reach: [f32; 2]` (the child's `[r_xz, r_y]`
  from its AABB, computed by `SdfNode::taper`); hand-written `Taper { .. }`
  literals must add it (`[f32::INFINITY; 2]` = no cone bound). Serialized
  trees from 1.x load with that default.
- `compiled::real::Real` is sealed (`f32` / `f32x8` only, as documented
  since 1.12.0).
- `CompiledSdf` fields are private: `instructions()`, `aux_data()`,
  `node_count()`, `lipschitz()`; `#[non_exhaustive]` removed.
- Optional dependencies are enabled with `dep:` — `--features wgpu` /
  `clap` / `pyo3` / `numpy` / `cranelift-*` / `pollster` / `bytemuck` /
  `futures-channel` no longer exist (use `gpu` / `cli` / `python` / `jit`);
  `image` stays a named feature (heightmap import). The `lazy_static`
  compatibility feature is gone.
- `optimize`: a taper with factor 0 is dropped as the identity (it used to
  drop factor **1**, which is not the identity).

### Changed — taper is a distance bound (law change, same shape)

- `Taper` returned the child's distance at the tapered point, which is
  not a parent-space distance: default tracing lost 4.9 % of the rays on
  the shrinking side. It now returns `real::taper_bound`: the child
  distance divided by the Jacobian norm over a ball (the map is a
  perspective projection with centre `(0, 1/f, 0)`; the norm grows
  towards that plane, so the ball is capped at half the distance to it),
  combined with the signed distance to the cone ∩ slab that contains the
  shape (from the child's reach). The second term is what keeps the plane
  `y = 1/f` from becoming a phantom surface — the Jacobian term alone goes
  to 0 there and *every* ray crossing the plane stopped on it (508 / 508 in
  the new `taper_singular_plane_is_not_a_surface`). Same law on the tree
  evaluator, compiled scalar / SIMD, interval arithmetic and the three
  shader helpers (`alice_taper_bound`); GPU parity and naga validation
  cover it. `tests/test_relaxed_tracing.rs`: taper moved from the pinned
  (6 %) to the exact set with four scenes, including ones whose singular
  plane lies inside the ray box. `eval_lipschitz` stays `INFINITY` for
  taper (no finite global constant). The BVH AABB of a taper is the cone's
  box (`r_xz (1 + |f| r_y)`), which the old `expand(extent · |f|)`
  undershot for shapes larger than 1.

### Changed — texture-fit uses the crate's PCG noise; GPU parity of the emitted shader

- The texture module had its own value noise (`fract(sin(dot) · 43758.5)`)
  duplicated in the emitted WGSL / HLSL / GLSL under the same
  `hash_noise_3d` name as the SDF transpilers' helper — a second law that
  no GPU reproduces exactly and that clashed when both shaders were pasted
  together. It now uses `modifiers::hash_noise_3d` (PCG lattice hash) on
  the CPU (scalar and SIMD) and embeds the transpilers' own helper text
  (`modifiers::HASH_NOISE_{WGSL,GLSL,HLSL}`, now public). **Fits made
  before this version reconstruct differently and must be regenerated.**
- Oracle `tests/test_texture_shader_gpu_parity.rs` (CI `gpu-parity` job):
  the emitted WGSL rendered through `GpuEvaluator` matches `reconstruct`
  to 3e-7 over a 4-octave result with rotated and axis-aligned octaves.

### Fixed — dual contouring fins at grid-tangent surfaces

- Where the surface is tangent to a grid plane (torus inner equator, sphere
  or cylinder radius on a plane) the cells on both sides see a sign change
  and both get a dual vertex; the quads between the two rows were thin fins
  folded whichever diagonal was chosen, and pointed inward.
  `triangulate_quads` now collapses a quad that folds on both diagonals
  along its shorter pair of opposite edges (a local edge collapse inside the
  cell). `tests/test_dual_contouring_invariants.rs` no longer skips small
  triangles and checks torus / sphere / cylinder at five resolutions.

## [v1.13.0] - 2026-09-16

Oracle tests for the paths that had none (dual contouring, non-Lipschitz tracing, SVO ray query, NPR colour laws, neural SDF, texture fitting, the Python binding) and the fixes they found, plus a local `scripts/preflight.sh` that reproduces every CI gate before a push.

### Fixed — texture-fit (found by the new oracle; the feature had no CI test step)

- The scalar noise (`hash_noise_3d_cpu`, `eval_octave`) had drifted from
  the SIMD lanes: the 1.12.0 clippy pass rewrote it with `mul_add`, and
  `fract(sin(dot) · 43758.5)` turns a 1-ulp difference in `dot` into a
  different corner value (`test_simd_matches_scalar` had been failing;
  no CI step ran the `texture-fit` tests). Scalar and SIMD now share one
  operation order, `#[allow(clippy::suboptimal_flops)]` with the reason.
- The fitter could not recover a texture that *is* one octave of its own
  law (NMSE 0.63 from a single Nelder-Mead start: the DCT band index is a
  coarse frequency estimate and the cost is periodic in phase). Each
  octave now scans 4 frequency scales × 4 phase quadrants with a short
  budget and refines the best start with the full budget; the same
  texture fits to NMSE 0.08 / 29 dB in one octave.
- Padded SIMD lanes of the subsampled cost (sample counts that are not a
  multiple of 8) contributed `amp² · noise(phase)²` each; masked out.
- DC bias accumulated in f32 (third decimal off on large images); f64.
- `TextureFitConfig::tileable` is documented as having no effect (the
  hash lattice does not wrap); `FrequencyBand::frequency` is documented
  as the DCT-II index, not cycles per image.
- Public: `texture::{reconstruct, eval_octave, nelder_mead, OptimizeResult}`.
- Oracle `tests/test_texture_fit_oracle.rs`: Nelder-Mead on a quadratic
  bowl / Rosenbrock / monotonicity, `eval_octave` vs an independent
  evaluation, noise range and continuity, flat image → bias only,
  synthesized octave recovered with reported PSNR ≡ PSNR of `reconstruct`
  and NMSE ≡ MSE / Var, more octaves never worse, determinism, padded vs
  aligned grid, and naga validation of the emitted WGSL / GLSL. CI runs
  the `texture-fit` lib tests and this oracle; clippy covers the module.
  Pending: a GPU parity run of the emitted shader against `reconstruct`.

### Added — Python binding smoke oracle in CI

- `python/tests/smoke.py` + ci.yml `python-smoke` job (`maturin develop
  --features python`, no pytest): every assertion has a closed-form
  answer — unit sphere / box distances, `eval_batch` ≡ |p| − 1,
  compiled ≡ tree, mesh vertices on the sphere with outward winding and
  volume within 5 % of 4/3 π, JSON / `.asdf` round trips. The binding had
  no CI coverage before (release-wheels only builds it).

### Changed — `lazy_static` compatibility feature, local CI preflight

- The FFI registries use `std::sync::LazyLock`; the `lazy_static` optional
  dependency is gone. Because an optional dependency is an implicit public
  feature, the name stays as an empty `lazy_static` feature that `ffi`
  still enables (cargo-semver-checks `feature_missing` /
  `feature_no_longer_enables_feature` are major breaks). Removed in 2.0.
- `scripts/preflight.sh`: every hard gate of ci.yml / security-audit.yml
  / fuzz.yml as the commands CI runs (actionlint, fmt, the three strict
  clippy sets plus an x86_64 cross-lint, MSRV 1.85 checks, feature builds,
  wasm32, rustdoc, semver-checks against crates.io, cargo-deny, machete,
  stub guard, fuzz build; `--quick` skips only the test suites). The
  pre-push hook runs it and blocks the push on failure; the semver break
  above reached CI because this file did not exist yet.

### Fixed — dual contouring (found by the new invariant oracle)

- **Every dual-contouring triangle was wound inward** (sphere res 32:
  0 outward / 3714 inward, signed volume −4.23): the orientation branch per
  edge axis was inverted, the mirror image of the marching-cubes finding.
  Non-planar quads are now split along the diagonal whose two triangles both
  face the quad's mean vertex normal (a fixed diagonal folded triangles on
  the inner ring of a torus), and "inside" is `d < 0` everywhere so a face
  lying exactly on a grid plane no longer produces in-plane quads with an
  arbitrary orientation (the CSG-subtract box lost triangles that way).
  Oracle: `tests/test_dual_contouring_invariants.rs` — outward winding,
  closed, vertices on the surface, signed volume within 5 % of the analytic
  sphere / torus, and the sharp-feature property (box vertices on its faces,
  corners within a quarter cell, volume error below marching cubes').
  Known residue: ~0.5 %-of-a-cell slivers where a surface is tangent to a
  grid plane (documented in the test, QEF clamping is backlog).

### Fixed — sparse voxel octree ray query (found by the new oracle)

- `SparseVoxelOctree::ray_query` sphere-traced with the raw node distance
  (sampled at the node centre, so up to a half diagonal too large inside
  the node) and declared a hit only at `|dist| < 0.001` on a
  piecewise-constant field — 59 of 256 rays through a CSG scene overshot
  and were lost. It now steps by `dist − half_diag(leaf)` (a safe bound for
  a 1-Lipschitz field), treats "within a half diagonal" as the surface
  lying in this leaf and locates the crossing by bisecting the sign of the
  query: 0 misses, every hit within two finest leaves of the analytic
  crossing. Oracle: `tests/test_svo_query_oracle.rs` (query error bounded
  by the leaf size derived from the subdivision rule, error decreasing with
  depth, ray query vs scan, linearisation preserving every node).
- `ffi` registries use `std::sync::LazyLock`; the `lazy_static` dependency
  is gone (the `ffi` feature keeps its name).
- CI: the strict clippy job lints every feature that builds on Linux (the
  crate-wide policy only covered the default + shader set before); the
  feature-gated SVO oracle runs in the test matrix.

### Changed — neural SDF default learning rate (found by the new oracle)

- `NeuralSdfConfig::default().learning_rate` is 1e-2 (was 1e-3). Measured
  against the analytic unit sphere on held-out points, the old default left
  the network at RMSE 0.26 after its 100 epochs (a quarter of the radius,
  11 % of points far from the surface with the wrong sign); 1e-2 reaches
  0.08 in the same time and 0.03 at 300 epochs. Oracle:
  `tests/test_neural_oracle.rs` (RMSE vs analytic, sign agreement, more
  epochs not worse, seed determinism, lossless save / load, `eval_batch` ≡
  `eval`).
- NPR colour laws: `tests/test_npr_analytic.rs` — exact toon band levels,
  ramp / rim / vignette / outline endpoints and monotonicity, posterize
  idempotence, palette endpoints, compiled pipeline ≡ closed-form
  composition (all laws were already correct).

### Added — tracing oracle for the non-Lipschitz laws

- `tests/test_relaxed_tracing.rs::non_lipschitz_laws_default_tracing`:
  domain repetition (`RepeatInfinite` / `RepeatFinite` / `PolarRepeat`) of a
  child symmetric inside its cell traces with 0 mismatches against the scan
  oracle (the common case is a distance field even though `eval_lipschitz`
  cannot prove it); an off-centre repeated child and a tapered box are
  pinned at documented miss-rate ceilings (taper over-estimates on its
  shrinking side, 4.9 % of rays — making the taper law a bound by dividing
  by the local Jacobian norm is backlog).

## [v1.12.0] - 2026-09-15

Bridge features back on crates.io (P15), crate-wide clippy pedantic + nursery policy, every CI test step a hard gate (blocking fuzz seed replay, semver-checks), colour-program stack validation, and the evaluator / marcher / law work that followed the 2026-09-15 maintainer self-review.

### Fixed — panics reachable from untrusted input

- `GpuColorProgram::deserialize` accepted an unbalanced stack program
  (every opcode decoded, operands missing), which then panicked in
  `CompiledColorPipeline::eval`. `ColorOp::stack_effect` is the single
  `(pops, pushes)` table, `CompiledColorPipeline::validate` simulates the
  stack, and `deserialize` returns `DeserializeError::Stack` for an
  unbalanced program; `eval` documents that it panics on one (programs from
  `compile` / `deserialize` never are). The remaining `unwrap` / `expect`
  sites in production paths were audited: infallible `write!` to `String`,
  slices with a checked length, documented-panic APIs with `try_` twins.
- `DeserializeError` and the new `StackError` are `#[non_exhaustive]`
  (error enums grow as validation improves). Adding `Stack` to the exhaustive
  `DeserializeError` is the one-time break this release accepts; the two
  corresponding cargo-semver-checks lints are downgraded to warnings in
  `Cargo.toml` (removed after the 1.12.0 publish) so the CI gate stays hard
  for everything else.
- Texture optimiser / spectrum sorts use `total_cmp` (a NaN cost no longer
  panics the sort).
- `Real` is documented as implemented for `f32` / `f32x8` only (required
  methods are added as laws need them; a private `Sealed` supertrait comes
  with 2.0).

### Changed — clippy policy: pedantic + nursery for the whole crate

- `Cargo.toml [lints.clippy]` now sets `pedantic` and `nursery` to warn
  with every exception listed and justified there (the former `lib.rs`
  allow list moved into it, so tests / benches / examples share the bar),
  and CI's clippy jobs run with `-D warnings`. Landing the policy fixed
  ~480 lib findings: `use_self`, `const fn`, `midpoint`, `mul_add` at 100
  sites (演算の掟 §2-4), a never-read Vec in the IGES writer, contains +
  insert, a decimal bitmask, `hypot`, an integer loop for the stairs step
  index, complete `Debug` impls; two `suspicious_operation_groupings` sites
  are documented false positives.

### Changed — CI gates (review R2-7 follow-up)

- `Test (AAA meta)` and `cargo-semver-checks` are hard gates (both were
  `continue-on-error`). The AAA step had been hiding `npr::scene_composer`
  tests that built GLSL / HLSL sources without those transpiler features;
  each test is now gated on the feature it needs.
- Fuzz: the committed regression seeds are replayed in a dedicated blocking
  step (a known crash regressing fails the job). Until now the seed branch
  never ran in CI — the path was spelled `fuzz/seeds/…` from inside
  `fuzz/`, so the directory test was always false. The time-boxed
  exploration run and the coverage job stay informational by design
  (documented in the workflows).

### Added — bridge features restored (roadmap P15)

- `physics` (alice-physics 1.1), `codec` (alice-codec 0.1.2), `asp`
  (libasp 1.0) and `sdf-cache` (alice-cache 0.2) resolve to the sibling
  crates on crates.io again — they were removed for the 1.7.7 publish while
  those crates were path-only. API drift since then was two `Result`s in
  the codec quantiser (buffers are sized to each other, so the error is
  impossible and is `expect`ed with that invariant) and a missing doc
  comment; the redundant `unsafe impl Send / Sync for CompiledSdfField` is
  gone (the type derives both). `font` stays an inert gate until alice-font
  publishes. New: ASP I-packet round-trip test.
- CI: a `bridges` job builds and tests each bridge feature on its own with
  the real crates.io dependencies, and the main test job runs all four
  together as a hard gate (that step was `continue-on-error` against
  features that did not exist). The `alice-stubs` action and every stub
  workaround in the workflows are removed — there are no path
  dependencies left to satisfy.

## [v1.11.0] - 2026-09-15

Maintainer self-review landing (two rounds, independent Linux x86_64 environment, 2026-09-15): every finding fixed with an oracle test on its path; the Lipschitz bound is applied by every marcher; seven primitive laws are now exact. Shape changes (Egg apex, Horseshoe legs, BlobbyCross arms) and the marching-cubes index order flip are listed under Fixed.

### Added

- `fuzz/fuzz_targets/fuzz_eval_parity.rs`: builds arbitrary primitive / CSG /
  modifier trees and asserts tree ≡ compiled scalar ≡ compiled SIMD at random
  points plus cell / sector / base-plane ties; `fuzz/seeds/<target>/` keeps
  every crash it found as a committed regression input that CI replays first.
- FFI: every exported `extern "C"` function (175) now runs through
  `ffi_guard`, which catches a Rust panic inside the call and returns the
  function's sentinel (null handle, `f32::MAX`, `0`, `false`,
  `SdfResult_Unknown`) instead of unwinding into the host — on Rust 1.81+ that
  unwind aborts Unity / Unreal / the Python interpreter. The message is kept
  per thread and read with the new `alice_sdf_last_error()` /
  `alice_sdf_clear_last_error()` (`include/alice_sdf.h` updated; no new
  `SdfResult` variant, the enum is exhaustive and 1.x stays semver-minor).
- `mesh::MeshInputError` and non-panicking `try_stripify`,
  `try_encode_index_buffer`, `try_encode_filter_oct_i16`,
  `try_decode_filter_oct_i16_in_place`, `try_encode_filter_quat_i16`; the
  existing functions keep their signature and forward to the `try_*` form
  (their `# Panics` contract is unchanged).

### Fixed — evaluation-path parity (found by the new `fuzz_eval_parity` target, 9 findings in its first hour)

- `Scale` / `ScaleNonUniform` in the compiled scalar, SIMD, BVH and JIT-SIMD
  paths multiplied every *leaf* distance by the factor; the tree, JIT-scalar
  and shader paths scale the blended result. The two agree only when every
  operator above the leaves is linear — `Scale(ExpSmoothUnion)` was 21% off,
  `Scale(SmoothUnion / Chamfer / Stairs / Round / Onion)` likewise. The scale
  is now applied when the `Scale` frame pops on every path.
- `Real::signum` is `x < 0 ? -1 : 1` on every path (`f32` used `f32::signum`,
  whose `-0.0 → -1` flipped the pyramid distance sign at its base centre
  against SIMD / JIT); the GLSL / WGSL / HLSL pyramid and hex-prism helpers
  emit the same conditional instead of `sign()` (0 at 0).
- Exponential smooth union / intersection / subtraction use the stable form
  `min(a, b) ∓ k·ln(1 + e^{-|a-b|/k})` on the CPU paths and in the shader
  text: the textbook `-k·ln(e^{-a/k} + e^{-b/k})` underflowed to `ln(0)` for
  `d ≫ k` (`+inf` on libm, NaN on the SIMD polynomial, `-ln(1e-10)·k` in the
  shaders' clamp).
- SIMD `atan2` is per-lane libm: the `wide` polynomial is a few ulp off and
  an odd polar-repeat count puts `atan2(0, -x) = π` exactly on a sector tie.
- Every repeat / polar / helix law is one function: the tree evaluator's
  `modifier_repeat_infinite` / `modifier_repeat_finite` / `modifier_polar_repeat`
  / `modifier_taper` and its `Rotate` arm now call the generic
  `compiled::real` laws (`p * (1 / s)` operand form, two-cross-product
  rotation) instead of keeping scalar copies that rounded differently by an
  ulp and crossed a cell / sector boundary (four nested polar repeats
  amplified it to a whole sector).
- `Taper` denominator `1 - f·y` is kept away from its singular plane
  (`|den| ≥ 1e-6`, sign preserved) on every path (the tree gave NaN, SIMD a
  finite value); the transpilers emitted `1 + f·y` — a *mirrored* taper — and
  multiplied the child distance by `den`, neither of which any CPU path does.
- Shader `RepeatFinite` clamped the cell index to `±count` instead of the CPU
  `±count/2` (twice the extent).
- `tests/test_gpu_law_parity.rs` (feature `gpu`, Metal-verified): taper,
  repeat, polar, pyramid / hex sign, scale-after-blend and exp-smooth laws
  agree with `eval` to 5e-7 relative on the GPU.

### Fixed — self-review 2026-09-15 (round 1: sphere tracing / Lipschitz / CI oracle)

- Over-relaxed sphere tracing (`RaymarchConfig::fast()` ω = 1.2,
  `RaymarchConfig::relaxed()` ω = 1.6, any `omega > 1`) never retreated: on an
  overshoot it advanced from the overshot position, left the ray behind the
  surface with `d < 0` and crawled `min_step` until `max_steps` — a unit
  sphere lost 76 % of its rays at ω = 1.6, a torus 86 %. The tree, compiled
  and JIT marchers (and `raymarch_detailed`, which ignored `omega`) now share
  one `RelaxedStepper` implementing Keinert et al. 2014: when the unbounding
  spheres of two consecutive samples do not overlap, or the sign of `d`
  flips, the ray retreats to the last safe point `t_prev + |d_prev| / L` and
  continues unrelaxed; the overshoot check runs before the `|d| < ε` hit test
  so a relaxed step that lands just past a thin feature is not reported as a
  hit. `raymarch_relaxed` / `raymarch_detailed` are now re-exported from
  `raycast` (they were unreachable dead code). Oracle:
  `tests/test_relaxed_tracing.rs` (24 × 24 rays vs a 0.5 mm scan + bisection,
  6 shapes × 4 configs × 3 paths, 0 hit/miss mismatches; relaxed tracing
  takes fewer steps than plain tracing at grazing incidence).
- `interval::eval_lipschitz` was unsound: the nine triply periodic minimal
  surfaces (Gyroid, Schwarz P, diamond, Neovius, Lidinoid, IWP, FRD,
  Fischer–Koch S, PMY) sat in the "exact SDF, L = 1" arm although they are
  implicit trigonometric functions with |∇F| up to 7 (Neovius) — so
  `RaymarchConfig::relaxed` stepped past their surface and Neovius / IWP could
  not be rendered; the noise / displacement bounds ignored the gradient of the
  offset field (`sin(5x)…` is ×5, sine displacement dropped its frequency,
  Perlin |∇| ≤ 3.5, value-noise fbm ≤ 3√3 per octave); chamfer / stairs /
  engrave (`(a + b)/√2`) and pipe (`√(a² + b²)`) are √2-Lipschitz, not 1; and
  the twist / bend factor assumed a radius of 10 with the wrong norm
  (`√(1 + v²)` instead of the shear singular value `v/2 + √(1 + v²/4)`) — it is
  now taken from the child's AABB (the plane / unbounded-child fallback keeps
  10). The bound is now defined on the exterior `{f ≥ 0}` (what sphere tracing
  needs) and every claim is analytic or a pinned numerical supremum; laws that
  are not Lipschitz there — domain repetition (`RepeatInfinite` / `RepeatFinite`
  / `PolarRepeat`) with an arbitrary child, `Taper` (singular plane),
  `ColumnsUnion` family and `LatticeDeform` (jumps), `HeightmapDisplacement`
  (dominant-axis switch), `SweepBezier` (nearest-parameter jumps), and the
  `Ellipsoid` / `Egg` / `Horseshoe` / `BlobbyCross` / `Stairs` / `Helix`
  primitives (their laws jump or grow unboundedly, see the follow-up entries)
  — return `f32::INFINITY` instead of a guess, and `RaymarchConfig::relaxed`
  falls back to plain tracing for them. Oracles:
  `lipschitz_claim_bounds_every_difference_quotient` (every corpus node,
  13 directions × 2 step sizes × 2600 points, difference quotient ≤ claim) and
  `lipschitz_claims_are_finite_where_the_law_is_lipschitz` in
  `tests/test_evaluator_opcode_parity.rs`; `tests/test_relaxed_tracing.rs::
  tpms_trace_correctly_with_lipschitz_bound` (six TPMS, 0 mismatches with the
  bound, Neovius / IWP demonstrably lose rays without it).
- `interval::eval_interval` returned `EVERYTHING` for the nine TPMS
  surfaces, so interval-based pruning silently did nothing on any scene
  containing one (review SDF-R2-5); they now use the centre sample ± L·ρ
  with the Lipschitz constants above (finite and sound, pinned by
  `tpms_intervals_are_finite_and_sound`).
- `RaymarchConfig::min_step` is now applied in field units (divided by
  `lipschitz` like the step itself). A floor in ray units moved a sample by up
  to `L·min_step` in field value and, for `L·min_step > ε`, carried it across
  the `|d| < ε` hit band into the interior (Neovius at L = 7 lost 16 % of its
  rays that way even with the correct bound). With `min_step ≤ epsilon` plain
  tracing can no longer overshoot at all.

### Added — Lipschitz bound applied by every marcher

- `CompiledSdf` is now `#[non_exhaustive]` (construct it with `compile` /
  `try_compile`; its fields stay readable). The new `lipschitz` field would
  otherwise have broken an exhaustive struct literal — no known consumer
  builds one, since `instructions` comes from the compiler — and the
  attribute keeps later fields semver-minor. Related: `Real` staying
  unsealed until 2.0 is tracked in the roadmap.

- `CompiledSdf::lipschitz` / `JitCompiledSdf::lipschitz()` record
  `eval_lipschitz(node)` at compile time, and `RaymarchConfig::with_bound`
  raises a configuration's `lipschitz` to a finite bound. The config-less
  entry points (`raymarch`, `raymarch_batch*`, `render_depth`,
  `render_normals`, `raymarch_compiled`, `raymarch_simd_8`, `raymarch_jit*`,
  the `*_with_config` compiled / JIT variants) now step by `d / L`, so a
  TPMS surface (L = √3 … 7) traces correctly without the caller knowing it
  is not a distance field: Neovius / IWP / Gyroid report 0 false hits
  against the scan oracle on every path (`default_entry_points_apply_the_
  lipschitz_bound`), where the plain `t += d` lost most rays. Trees with no
  finite bound keep stepping by `d`. `raymarch` on a tree computes the bound
  per call (a tree walk; twist / bend children cost an AABB pass) — the batch
  / render functions compute it once and the compiled marchers not at all.

### Fixed — laws that were not distance fields (found by the Lipschitz property test)

- `Egg` was a three-branch approximation that reported **positive**
  distances for interior points on the axis (`egg(1, 0.5)` at (0, 0.1, 0) =
  +0.9) and jumped by `ra − rb` across the origin. It is now Inigo Quilez's
  exact `sdEgg` (disc of radius `ra` below y = 0, arcs of radius
  `2(ra − rb)`, apex cap `rb` at `y = √3(ra − rb) + ra`) on the CPU and in
  the WGSL / GLSL / HLSL helpers (GPU ↔ CPU 4.2e-7); `eval_lipschitz` is 1
  again and the interval bounding sphere covers the apex. **Shape change**:
  the apex moved from `y = ra` to `y = √3(ra − rb) + ra`.

- `Horseshoe` mixed an `abs(qx)` leg mirror with the width / thickness
  terms and was not a distance field (difference quotients up to √2). It
  is now IQ's exact `sdHorseshoe` (band of half-width `width` around an
  arc of `radius` opened by `angle`, legs of `half_length`) extruded by
  `thickness`, mirrored in the three shader helpers (GPU ↔ CPU 3.0e-7);
  `eval_lipschitz` is 1 again. **Shape change**: legs now end flat at
  `half_length` and the band is symmetric about its centre line.

- `BlobbyCross` was a home-grown "sqrt blend" that jumped by up to 23× the
  sample spacing between its two regions. It is now IQ's exact
  `sdBlobbyCross` (nearest parameter on the parabola arms from the
  depressed cubic, `he = 0.5`) on `|xz| / size`, extruded along Y; CPU and
  the three shader helpers are mirrored (GPU ↔ CPU 8e-5, `pow` / `acos`
  domain), the interval arm is the exact-SDF form and `eval_lipschitz` is
  1. **Shape change**: arms are parabola segments reaching `±size`.

- `SweepBezier` found the nearest curve parameter with 5 samples + Newton on
  the CPU / SIMD paths and with 4 Newton steps from `t = 0.5` in the
  shaders — three different laws, all jumping between local minima
  (difference quotients 800× the sample spacing). The distance is now IQ's
  closed-form `sdBezier` (Cardano / trigonometric cubic roots, degenerate
  curve → segment) in `modifiers::sweep::bezier_distance_2d`, used per lane
  by the SIMD path and emitted as one `bezier_distance_2d` helper per shader
  language (GPU ↔ CPU 4.3e-5); `eval_lipschitz` is the child's bound.

- `Stairs` compared only the step boxes {si − 1, si, si + 1, sj} and so
  missed the nearest step for points above or beside the staircase, jumping
  by up to 84× the sample spacing where the candidate set changed. It now
  takes the exact minimum over all `n_steps` boxes (CPU and the three
  shader helpers, GPU ↔ CPU 2.4e-7); `eval_lipschitz` is 1.

- `Helix` measured the distance to the helix point at the query's own
  azimuth (an over-estimate, and undefined on the axis, where the field
  jumped by 15× the sample spacing). It now finds the true nearest curve
  point by Newton from the same-azimuth candidates of the three nearest
  wraps (brute-force agreement 2e-3 over three pitch / radius ratios,
  continuous on the axis, azimuth pinned to 0 there because GPU `atan2(0, 0)`
  is NaN); the three shader helpers mirror it (GPU ↔ CPU 1.7e-6) and
  `eval_lipschitz` is 1.

- `Ellipsoid` was Inigo Quilez's `k0·(k0 − 1)/k1` approximation: not a
  distance bound (its gradient grows like `(max r / min r)⁴` far from the
  surface, so sphere tracing could skip an anisotropic ellipsoid) and
  discontinuous at the centre. It is now the exact signed distance
  (Eberly's robust nearest-point algorithm: axes sorted, point folded into
  the first orthant, bisection for the Lagrange parameter with the
  lower-dimensional reductions for axis-plane queries), on the CPU, per
  lane in the SIMD path, and as one `sdf_ellipsoid` helper per shader
  language (GPU ↔ CPU 3.3e-7 including on-axis queries and a 10:1 flat
  ellipsoid); brute-force agreement 4e-3 (sampling-limited) and
  `eval_lipschitz` is 1. Cost: up to 64 bisection steps per evaluation on
  this primitive only.

### Changed — compiled evaluator speed (review SDF-R2-4)

- `eval_compiled` zero-filled its three evaluator stacks (≈ 3.4 KB for f32,
  ≈ 6 KB for the SIMD lanes) on every call, a ≈ 21 ns fixed cost that made
  the "recommended" scalar VM slower than the tree walker for anything
  under ~30 nodes (sphere 30 ns vs 8.6 ns, 20-node CSG 82 ns vs 63 ns). The
  stacks are now uninitialised slots written before they are read (the
  stack discipline of the bytecode; debug builds assert every read):
  sphere 4.0 ns, 20-node CSG 58 ns. The SIMD batch path shares the gain.
- `sdf_to_mesh` / `marching_cubes` evaluate the grid through the compiled
  SIMD batch evaluator (tree ≡ compiled by the parity corpus), falling back
  to the tree walker only for trees the compiler rejects: the 20-node scene
  at res 128 goes from 60 ms to 38 ms (bench `marching_cubes/complex`).

### Fixed — self-review 2026-09-15 (transpiler validation, found by the new naga oracle)

- The five GDF polyhedra (`Tetrahedron`, `Dodecahedron`, `Icosahedron`,
  `TruncatedOctahedron`, `TruncatedIcosahedron`) transpiled to a call of
  `sdf_<name>(...)` that no transpiler defined — every WGSL / GLSL / HLSL
  shader containing one failed to compile on the GPU. The helpers are now
  emitted in all three languages, mirroring `primitives::gdf_vectors`
  (GPU ↔ CPU ≤ 3.4e-7 on Metal).
- `ColumnsUnion` emitted a truncated declaration (`var d3_a2 = mi    var
  d3_m = …`) left behind by an abandoned string-building attempt; the arm
  now emits the law once. Its modulo was WGSL `%` / HLSL `fmod` (truncated,
  sign of the dividend) while the CPU law and GLSL `mod` are floor modulo —
  a whole column period of drift for negative operands; `modulo_expr` is
  floor modulo in every language.
- GLSL `PolarRepeat` emitted `atan2(y, x)`, which GLSL does not have
  (`atan(y, x)`); the walker now goes through `ShaderLang::atan2_expr`.
- Each transpiler's `generate_shader` kept a second, hand-copied table of
  helper sources that silently skipped unknown names (`_ => {}`) — that is
  how the polyhedra went missing. The single `helper_source` table is now
  the only source and an unregistered helper panics at transpile time.
- GPU marching cubes failed to build its Pass 3 pipeline on DX12 (Windows,
  FXC `X4505: Sum of temp registers and indexable temp registers exceeds
  limit of 4096`): the 4096-entry triangle table was a WGSL module `const`
  that naga's HLSL backend lowers into indexable temporaries. It is now a
  read-only storage buffer (`TRI_TABLE`, binding 5). Hidden until now by
  the `continue-on-error` on the shader test step.
- Oracle: `tests/test_transpiler_naga_validate.rs` parses **and validates**
  (`naga::valid::Validator`) the WGSL and GLSL of every corpus node
  (features `gpu` / `gpu,glsl`); the corpus moved to `tests/common/corpus.rs`
  so every integration test can share it.

### Fixed — self-review 2026-09-15 (round 2: marching cubes output)

- **Every marching-cubes triangle was wound inward** (CPU `marching_cubes` /
  `sdf_to_mesh`, the compiled and adaptive variants, and the GPU compute
  path): `CORNER_OFFSETS` numbered the cube with its second and third axes
  swapped relative to the Bourke / Lorensen edge and triangle tables, a
  mirror image of the table's cube. A unit sphere at res 32 had 3608 of 3608
  triangles facing inward and a signed volume of −4.088 (truth +4.189); STL
  facets stored the (correct) averaged vertex normal next to a contradicting
  vertex order, so slicers that use the winding read every export as an
  inside-out solid. The corner numbering now matches the tables (0–3 on the
  y = 0 face, 4–7 on y = 1) on both CPU and GPU. **Breaking for consumers
  that compensated for the flip** (e.g. rendered with front-face culling set
  to CW, or negated normals from `(b − a) × (c − a)`): mesh topology and
  vertex positions are unchanged, only the index order per triangle.
- `sdf_to_mesh` was not closed on grids aligned with the surface: a grid
  edge shared by four cells was interpolated from each cell's local endpoint
  order, so the four copies differed in their last bits and vertex
  deduplication could not merge them (16–64 open edges at res ≥ 64). Edge
  vertices are now interpolated from the lexicographically smaller corner in
  every cell (bit-identical), and triangles that collapse when a grid corner
  sits exactly on the iso-level are dropped after deduplication
  (`mesh::remove_degenerate_triangles`, also exported). Oracle:
  `tests/test_mesh_orientation.rs` — all triangles outward against ∇f,
  signed volume positive and within 5 % of the analytic sphere / torus,
  zero open edges, vertex count = distinct positions, STL round trip facet
  normal ∥ winding; the GPU variant runs under `--features gpu-mesh`
  (Metal-verified).

### Changed

- FFI handle registries tolerate a poisoned lock (`PoisonError::into_inner`):
  a caught panic while a registry lock was held no longer turns every later
  FFI call into an error.
- CI: the `unity` / `unreal` meta-feature builds are hard gates
  (`continue-on-error` removed).
- CI: new `gpu-parity` job (ubuntu + Mesa lavapipe software Vulkan) runs
  the GPU ↔ CPU law / noise parity tests, the naga shader validation and
  the GPU marching-cubes orientation test with `ALICE_SDF_REQUIRE_GPU=1`,
  which turns "no adapter → skip" into a failure. Until now the only
  correctness oracle for the transpilers had never executed in CI; the
  `Test (shader transpilers)` step is also a hard gate (its
  `continue-on-error` is removed).

## [v1.10.3] - 2026-09-15

### Fixed

- Cell-boundary rounding now agrees on every evaluation path. The repeat /
  polar-repeat / helix laws snapped with `round`, whose tie direction differs
  per path (`f32::round` ties away from zero; `wide::f32x8::round` ties to even
  on AVX / NEON but away from zero on the SSE2 fallback; Cranelift `nearest`,
  WGSL `round` tie to even; GLSL `round` is implementation-defined; HLSL
  `round` ties away). A point on a cell boundary — every marching-cubes grid
  whose step divides the spacing has them — was folded into a different cell
  per path, changing the distance by a whole cell (1.2 for
  `sphere(0.3).translate(0.6,0,0).repeat_infinite(2,2,2)` at x = ±1: scalar
  1.3 / SIMD 0.1 / JIT 1.3). The canonical rule is now `floor(x + 0.5)`
  (`crispy::round_half_up`, `Real::round_half_up`) in the tree evaluator, the
  generic scalar / SIMD stack machine, the interval evaluator, both JITs and
  the GLSL / WGSL / HLSL transpilers. `Real::round` is unchanged but documented
  as not path-safe at ties.
- `PolarRepeat` tree evaluation used a different law from the compiled paths
  (`%` fold with a `+100·sector` offset) and picked a different sector at exact
  sector boundaries; it now delegates to the same `sector` / `count / TAU`
  round-trick law the compiler bakes into the instruction, and the shader
  transpilers snap with `angle * (n / TAU)` (same operands) instead of
  `angle / sector`.
- Parity corpus: tie sample points and offset (asymmetric) repeat children
  added (`tests/test_evaluator_opcode_parity.rs`); new
  `tests/test_round_tie_parity.rs` pins tree / compiled / SIMD / JIT agreement
  at cell boundaries and asserts the shader text uses `floor(x + 0.5)`.
- Test gating: `tests/noise_shader_validate.rs` and the two WGSL
  `npr::scene_composer` tests require the `gpu` feature (they use
  `WgslShader`) and are now gated on it, so `--features glsl,hlsl` without
  `gpu` compiles and passes.

### Changed

- `rust-version` corrected from `1.75` to `1.85`: the declared MSRV had been
  false since the lockfile picked up `clap_lex 1.1.0` (edition 2024), so
  `cargo check` on 1.75 failed at manifest parsing for the default `cli`
  feature. 1.85 is verified for the default and docs.rs feature sets; a CI
  `msrv` job now pins the declared toolchain.
- docs.rs builds with `glsl, hlsl, jit, svo, terrain, destruction, gi, ffi`
  (`[package.metadata.docs.rs]`); previously only the default feature was
  documented, hiding the transpilers, the JIT and the AAA modules.
- Cargo.toml `description` no longer references 1.7.7 / 1.8.0 for the bridge
  features (they remain path / git only).
- CI: strict clippy runs with `--all-targets` (tests, benches, examples).
- `resolver = "3"` (MSRV-aware dependency resolution, cargo 1.84+): `cargo
  update` no longer selects dependencies whose `rust-version` exceeds the
  crate's, so the lockfile cannot drift away from the declared MSRV again.
- Two `Option::map_or(true, ..)` sites rewritten as `is_none_or` — clippy's
  `unnecessary_map_or` had been silenced by the false 1.75 MSRV.

## [v1.10.2] - 2026-09-14

### Fixed

- `SdfNode` now drops iteratively (`src/types/drop.rs`): children are moved onto an explicit heap stack and released one `Arc` at a time, so freeing a tree no longer recurses once per level. A 2,400-deep `subtract` nest (the shape ALICE-LOL's stdlib products produce) overflowed a 2 MB thread stack on drop; the regression test now builds and drops 100,000-deep chains in a 256 KB thread. Shared subtrees (`Arc::clone`) are left to their last owner as before. Recursive `clone` / `node_count` / evaluators are unchanged.
- crates.io metadata: `homepage` / `documentation` (docs.rs) added to `Cargo.toml` — the crates.io page had no Documentation link before 1.10.2 (published 2026-09-15 together with 1.10.0 / 1.10.1 changes).
- README: bridge-feature notes no longer reference "v1.7.7 / v1.8.0"; bridges remain unavailable on crates.io releases (1.7.7 → 1.10.x).

## [v1.10.1] - 2026-09-14

### Changed — 1.10 Phase 2 (G1 + G2): one law per basic primitive and CSG operator

- The 13 basic primitives (`sphere` / `box3d` / `cylinder` / `torus` / `plane` / `capsule` / `cone` / `ellipsoid` / `rounded_cone` / `pyramid` / `octahedron` / `hex_prism` / `link`) and all 24 CSG binary operators (`union` … `tongue`, smooth / chamfer / stairs / columns families, `exp_smooth_*`) now have a single generic body `sdf_x_r<R: Real>` in `primitives::*` / `operations::*`. The existing scalar functions (`sdf_sphere(Vec3, f32)` etc.) are unchanged in signature and delegate to the generic law; the scalar and SIMD evaluator tables call the same generic function. Branches became `Real::select` (`cone`, `rounded_cone`, `pyramid`, `octahedron`, `ellipsoid` centre, `columns` early-out), `hypot` became `sqrt(x²+y²)` (≤ 1 ulp difference).
- Removed the SIMD-only `smooth_min_simd_rk` / `chamfer_min_simd` / `stairs_min_simd` / `eval_per_lane_binary` helpers (the generic laws replace them; `columns_*` and `exp_smooth_*` are now SIMD-native instead of per-lane).
- Added `Real::signum`.

## [v1.10.0] - 2026-09-14

### Changed — 1.10 Phase 1: one stack machine for scalar and SIMD

- **`compiled::real::Real`** — new scalar abstraction implemented for `f32` and `wide::f32x8` (`sqrt` / `abs` / `floor` / `round` / `min` / `max` / `sin_cos` / `atan2` / `exp` / `ln` / comparisons + `select` / per-lane `map` escape hatches) with `Vec3R<R>`. Every transform, modifier and post-processing law of the bytecode evaluator now exists once, as a generic function in `compiled::real` (`rotate_inverse` / `twist` / `bend` / `repeat_*` / `elongate` / `mirror` / `octant_mirror` / `revolution` / `extrude_*` / `taper` / `polar_repeat` / `shear` / `sweep_bezier` / `exp_smooth_*`), unit-tested against the canonical `modifiers::*` laws on both instantiations.
- **`compiled::eval_core::eval_bytecode<R>`** — the stack machine is generic; `eval_compiled` is `eval_bytecode::<f32>`, `eval_compiled_simd` is `eval_bytecode::<f32x8>`, `eval_compiled_bvh` shares the `f32` instantiation. The 2,250-line hand-written SIMD evaluator and the 1,400-line scalar evaluator are gone; the only per-instantiation code left is the leaf-primitive / CSG-binary law table (`compiled::prim_table::PrimTable`, bodies moved verbatim into `prim_table_scalar.rs` / `prim_table_simd.rs`). Phase 2 folds those into generic `sdf_x<R: Real>` laws.
- SIMD `LatticeDeform` now evaluates the lattice once per lane instead of twice (the per-lane escape returns point and Jacobian together).
- Performance (Apple Silicon, A/B against 1.9.2, min of 3 interleaved rounds): the generic evaluator is faster on every compiled path — `sphere` 38.4 → 30.2 ns (−21%), `translate×5` −17%, `rotate×5` −11%, `round×5` −13%, `twist×5` −9%, BVH sparse scene −5…−11%, 8-lane SIMD `translate×5` −9%, `twist×5` −23%, SoA 10k −5%. Getting there required three fixes worth recording: push frames as a direct struct literal (no temporary), leave the value stack untouched at `PopTransform` for point-only frames, and keep per-lane frame data (Extrude's z, LatticeDeform's Jacobian) in a side array instead of inflating every frame by two `R`s. New `benches/frame_cost.rs` guards exactly this.
- Public API unchanged: `eval_compiled*`, `eval_compiled_simd`, `eval_compiled_batch_simd(_parallel)`, `eval_gradient_simd`, `eval_distance_and_gradient_simd`, `Vec3x8`, `Quatx8`, the SoA entry points and the BVH entry points keep their signatures. `compiled::real` is public so downstream code can write `Real`-generic SDF laws.

### Fixed

- `Noise` evaluated Perlin gradient noise on the CPU / SIMD / bytecode paths but the shader transpilers emitted `hash_noise_3d` value noise. The transpilers now emit `perlin_noise_3d` — a verbatim port of `modifiers::perlin_noise_3d` (xor-multiply lattice hash of the `i32` cell coordinates and seed, 16-entry gradient LUT, quintic fade) — in GLSL / WGSL / HLSL, so every path renders the same Perlin field. `tests/test_gpu_noise_parity.rs` (feature `gpu`) measures |GPU − CPU| ≤ 7.2e-7 for `Noise` as well; `tests/noise_shader_validate.rs` (feature `glsl`) parses the generated WGSL and GLSL with naga. Shader-rendered `Noise` patterns change from value noise to the intended Perlin.
- `SurfaceRoughness` evaluated a different noise on every path: the CPU used its own `sin`-hash value-noise fbm (amplitude 0.5, per-octave rotation) while the GLSL / WGSL / HLSL `hash_noise_3d` helpers used a `sin`-hash fbm with amplitude 1.0 and no rotation, and `sin` of large arguments differs between GPU and CPU anyway. The noise law now exists once: `modifiers::surface_roughness::{hash_noise_3d, fbm}` (PCG lattice hash over the corner bits + seed, trilinear smoothstep blend, `2v - 1`; fbm = `Σ 0.5^i · noise(p · 2^i, 42)`) and all three shader helpers emit the identical function with `floatBitsToUint` / `bitcast` / `asuint`. `tests/test_gpu_noise_parity.rs` (feature `gpu`, skips without an adapter) measures |GPU − CPU| ≤ 7.2e-7 on 2048 points. The interval widening is `|amplitude| · (2 − 2^(1 − octaves))`. Realised roughness patterns change (statistics unchanged); the `Noise` node still pairs CPU Perlin with shader value noise and is tracked separately.
- `interval::eval_interval` violated its own contract (`lo ≤ sdf(p) ≤ hi`) for 32 node kinds — 2,045 violations on the 120-node parity corpus. Bounding-sphere lower bounds assumed a radius the shape exceeded or an exact distance the evaluation does not provide (`RoundedCylinder`, `HexPrism`, `Tube`, `TriangularPrism`, `Tetrahedron` / `Dodecahedron` / `Icosahedron` / `Truncated*`, `BoxFrame`, `Ellipsoid`, the extruded 2D shapes, …), `Taper` / `Shear` / `RepeatFinite` / `OctantMirror` / `IcosahedralSymmetry` / `LatticeDeform` mis-modelled their domain maps, and `Chamfer*` / `Stairs*` / `ExpSmooth*` under-estimated the blend widening. Arms now use a Lipschitz form (`eval(centre) ± L·half_diagonal`, sound for every 1-Lipschitz law), exact interval arithmetic on the actual formulas (`Superellipsoid`, `BlobbyCross`, `Tunnel`, `Helix`, chamfer), or interval images of the domain maps. `tests/test_evaluator_opcode_parity.rs::interval_eval_contains_point_values` pins the contract on the whole corpus; `analytic_gradient_matches_numerical` does the same for `eval::gradient` (already sound).
- `modifiers::surface_roughness` hash was `fract(sin(h)) * 43758` (±43758) instead of `fract(sin(h) * 43758)`, and `value_noise` hid it with a trailing `fract()` that turned the "smooth value noise" into a discontinuous sawtooth in (-1, 1); fbm therefore left `[-1, 1]` and the roughness interval was unsound. Fixed at the source (GLSL-style non-negative fract, no wrap); the shader paths already used their own `hash_noise_3d` fbm and are unaffected.
- `examples/gpu_eval.rs` did not build with `--features gpu` (`WgslShader::transpile` gained a `mode` argument). CI now builds the feature-gated examples (`cargo build --examples --features "glsl,hlsl,gpu"`) so example API drift fails fast.

## [v1.9.2] - 2026-09-14

**One law per opcode, every path** — closes the follow-ups left open by 1.9.1: the shader / JIT backends now share the CPU sign convention, the BVH is an annotation pass over the main compiler instead of a third hand-copied compiler, the JIT SIMD compilers fail loudly instead of emitting `f32::MAX`, and the parity corpus covers six evaluation paths plus an AABB-conservativeness oracle. First crates.io release since 1.9.0 (1.9.1 was never published; its notes below are included).

### Fixed

- **`Plane` sign in GLSL / WGSL / HLSL / BlinkScript transpilers and the Cranelift JITs** (`dot(p, n) + d`) now matches `sdf_plane` and every CPU evaluator (`dot(p, n) - d`, "distance from origin"). **Shader output changes for scenes using `Plane`** — flip the sign of `distance` if you relied on the old convention.
- **`sdf_regular_polygon` was unbounded**: the sector fold only handled `z >= 0` and the 2D term was a half-plane distance, so half the plane evaluated as "inside". Replaced with the exact regular-polygon law (circumradius `radius`, XZ plane) in Rust and in all three shader helper libraries. Caught by the new AABB-conservativeness test.
- **`LatticeDeform` outside its bounding box**: points were clamped to the boundary (every outside point mapped to the same deformed point) and the `0.1` Jacobian floor inflated distances 10x. Outside points now pass through unchanged with correction `1.0` (standard FFD). Tree result changes for scenes evaluating `LatticeDeform` outside the lattice.
- **BVH compiler was a third hand-copy of the compile law** (68 arms, no AABB laws for 60+ primitives, so it rejected them). `CompiledSdfBvh::try_compile` now reuses `CompiledSdf::try_compile` and computes AABBs with the instruction-driven walker in `compiled::refit` — every tree the main compiler accepts, the BVH accepts (the nine kinds 1.9.1 rejected included). `refit::primitive_aabb` / `csg_binary_aabb` / `transform_or_modifier_aabb` are exhaustive over `OpCode`.
- **`refit` scene AABB** was the union of every intermediate push (an inner child's untransformed bound leaked into the scene bound after `refit_all`); it is now the root's AABB.
- **`refit` `RoundedCone` AABB** ignored the end-cap spheres (`y` range is `[-hh - r1, hh + r2]`).
- **JIT SIMD (`jit::JitSimdSdf` / `JitSimdSdfDynamic`) silently pushed `f32::MAX` for the 82 opcodes without a codegen arm** and treated `Noise` as a no-op. `compile` now returns `Err("JIT SIMD: no codegen arm for opcode …")` so callers fall back to the interpreter instead of rendering nothing.
- **Cranelift JIT law drift** caught by the extended corpus: `Ellipsoid` centre (`0` instead of `-min(radii)`), `Engrave` (`0.5` instead of `1/√2`), `RepeatFinite` (clamped to `±count` instead of `±count/2`), `Bend` rotation sign, and unreduced degree-5 Taylor sin/cos (>10% error for `|x| > 2` in `Twist` / `Bend` / `PolarRepeat`) — both JITs now use range reduction to `[-π/2, π/2]` plus degree-9/8 series (≤ 3e-5 abs error).

### Added

- `compiled::OpKind` + `OpCode::kind()` — exhaustive stack-machine classification (`Primitive` / `Binary` / `Transform` / `Modifier` / `PopTransform` / `End`); `is_primitive` / `is_binary_op` / `is_transform` / `is_modifier` are now derived from it instead of numeric ranges.
- `CompiledSdfBvh::aux_data` — the BVH now carries the side buffer (heightmaps, lattices, bones, IFS matrices, polygon vertices). **Struct-literal constructors of `CompiledSdfBvh` outside the crate must add the field.**
- `tests/test_evaluator_opcode_parity.rs` now compares six paths (tree / scalar / SIMD / BVH / Cranelift JIT / JIT SIMD, the last two under `--features jit`) and adds `primitive_and_scene_aabbs_are_conservative` (grid-samples every corpus node: `sdf(p) ≤ 0 ⇒ p ∈ scene AABB`).

### Changed

- `compiled::jit_simd::JitSimd` is **deprecated** and is now a thin wrapper over `compiled::jit::JitSimdSdf` (it was a 2,500-line divergent copy). `compile` / `eval` / `eval_soa` keep their signatures.
- `refit::RefitError::UnsupportedOpcode` is no longer produced (every opcode has an AABB law); the variant is kept for API compatibility.

## [v1.9.1] - 2026-09-14

**Compiled evaluator parity** — every compiled evaluation path now executes the same law as the tree evaluator, and the opcode dispatch is exhaustive by construction.

### Fixed

- **Scalar / BVH evaluators silently mis-evaluated 11 / 18 opcodes.** `eval_compiled` lacked arms for `Circle2D` / `Rect2D` / `RoundedRect2D` / `Segment2D` / `Polygon2D` / `Annular2D` / `ExpSmoothUnion` / `ExpSmoothIntersection` / `ExpSmoothSubtraction` / `Shear` / `Animated`; `eval_compiled_bvh` additionally lacked `IFS` / `SdfSkinning` / `LatticeDeform` / `HeightmapDisplacement` / `IcosahedralSymmetry` / `ProjectiveTransform` / `SurfaceRoughness`. A `_ =>` fallback evaluated unknown primitives as a unit sphere, unknown binary ops as plain `min`, and unknown modifiers as a no-op (which for `Shear` then underflowed the coordinate stack at `PopTransform`). Both evaluators are now thin wrappers over a single exhaustive stack machine (`compiled::eval_scalar_core`), so a new `OpCode` without an evaluator arm is a compile error.
- **`Polygon2D` lost its vertices at compile time** and evaluated as a unit sphere on every compiled path. The compiler now serialises vertices into `aux_data` and all paths evaluate the real polygon.
- **`SineDisplacement` lost its frequency at compile time** (collapsed to the legacy `Displacement` law with frequency 5). `Instruction::displacement` now carries `[amplitude, fx, fy, fz]` and a new `Instruction::sine_displacement` preserves the per-axis frequency.
- **`LatticeDeform` ignored the Jacobian correction** on compiled paths; the tree law `eval(child, q) / correction` is now applied at `PopTransform` (per lane on SIMD).
- **BVH compiler silently replaced unsupported nodes with `sphere(0.001)`** (`IFS` / `SdfSkinning` / `LatticeDeform` / `HeightmapDisplacement` / `IcosahedralSymmetry` / `ProjectiveTransform` / `SurfaceRoughness` / `SineDisplacement` / `Terrain`). `CompiledSdfBvh::try_compile` now rejects them with `CompileError::UnsupportedPrimitive`; `BvhCompiler::compile_node` is exhaustive.
- **BVH compiler dropped `OctantMirror`** (compiled the child only). It now emits the modifier with a symmetric cube AABB.
- **`CompiledSdf` compiled `Terrain` to a silent sphere**; `try_compile` now rejects it with `UnsupportedPrimitive("Terrain")`.
- **SIMD (`eval_compiled_simd`) diverged from the scalar law** on: `Plane` (`dot + d` instead of `dot - d`), `Ellipsoid` (0 instead of `-min(radii)` at the centre), `RepeatFinite` (clamped to `±count` instead of `±count/2`), `HeightmapDisplacement` (added instead of subtracted the displacement, sine fallback when aux missing), `SurfaceRoughness` (ad-hoc FBM instead of `modifiers::surface_roughness`), `Segment2D` / `Polygon2D` (bounding-sphere fallback), `ExpSmooth*` (Schraudolph exp + a Padé ln whose linear term was ~2x off → up to 35% error), and every trig-based opcode (Bhaskara I sin/cos with 1.6e-3 abs error, 3.8e-3 rad atan2 — amplified to >10% on TPMS / `Twist` / `Bend` / `PolarRepeat`). SIMD now uses `wide`'s `sin` / `cos` / `atan2` and calls the shared scalar law per lane where no exact SIMD form exists.

- **`--features font` (and therefore `--all-features`) did not build on crates.io.** The empty `font` feature gated `font_bridge`, which imports the undeclared `alice_font` crate, so every user enabling it — and `cargo-semver-checks`, which enables all features — hit `unresolved import alice_font`. The feature is kept (removing it would be a semver-major break) but is now inert: `font_bridge` and the `text_to_3d_demo` example additionally require `--cfg alice_font_bridge` plus a local `alice-font` path dep.
- **`godot` feature did not build**: `to_glsl` called a non-existent `compiled::glsl::transpile_glsl`; it now uses `GlslShader::transpile(node, GlslTranspileMode::Hardcoded).source`.
- **`MeshRepair::repair_all` left non-manifold edges behind.** `merge_duplicate_vertices` can collapse two vertices of a sliver triangle (new zero-area face) and make two neighbouring slivers reference the same three vertices (duplicate face → edge with 4 incident triangles). The old order (`degenerate → merge → fix_normals`) never cleaned what the merge created — a marching-cubes `sphere(10)` at resolution 96 kept 96 non-manifold edges after repair (measured 2026-09-14 in the text-to-print 3MF export path). `repair_all` now runs `degenerate → merge → degenerate → remove_duplicate_triangles → fix_normals` and the sphere / table regression test asserts 0 non-manifold and 0 boundary edges.

### Added

- `MeshRepair::remove_duplicate_triangles` — drops winding-insensitive duplicate faces and index-collapsed triangles (keeps the first occurrence).
- `SdfNode::box3d_half_extents(hx, hy, hz)` — half-extent spelling of the plain box, matching `rounded_box` and the LOL DSL `box3d`. `box3d` (full dimensions) and `rounded_box` (half-extents) docs now state the asymmetry explicitly; neither signature changes.
- `primitives::{sdf_circle_2d, sdf_rect_2d, sdf_rounded_rect_2d, sdf_segment_2d, sdf_annular_2d, sdf_polygon_2d, sdf_polygon_2d_xy, sdf_polygon_2d_flat, extrude_2d}` — single-source 2D-extruded primitive laws used by tree / scalar / BVH / SIMD.
- `operations::{sdf_exp_smooth_union, sdf_exp_smooth_intersection, sdf_exp_smooth_subtraction}` — blend-width (`d/k`) exponential smooth laws (distinct from the rate-based `smooth_min_exp`).
- `modifiers::modifier_shear` — inverse shear law shared by all evaluators.
- `Instruction::sine_displacement(amplitude, fx, fy, fz)`.
- `tests/test_evaluator_opcode_parity.rs` — 120-node corpus covering every compilable `SdfNode` variant, compared across tree / scalar / SIMD / BVH at 8 sample points, plus a guard that the corpus reaches all 124 emitted opcodes and that unsupported nodes are rejected loudly.

### Changed

- `compiled::eval` and `compiled::eval_bvh` are now thin wrappers; the stack machine lives in `compiled::eval_scalar_core` (~2,300 lines of duplicated dispatch removed).
- `CompiledSdfBvh::try_compile` rejects the nine node kinds listed under Fixed (previously accepted and mis-compiled).

### Known limitations

- (both resolved in 1.9.2) The shader / JIT `Plane` sign and the `LatticeDeform` outside-bbox law.

## [v1.9.0] - 2026-09-13

**NPR compiled pipeline + SIMD batch + GPU bytecode** — Phase 12-D / 13 / 14 landing as additive minor bump on top of 1.8.0 NPR module foundation. Plus rustdoc broken-intra-doc-link fix and 6-issue clippy cleanup in test code.

### Added

- **Phase 14** — GPU bytecode serialisation and WGSL evaluator emitter. New public API:
  - `npr::compiled_color::gpu_opcode_tag` — stable `u32` tag constants for all 17 native opcodes (public so the WGSL evaluator's constants stay in lockstep with Rust)
  - `npr::compiled_color::gpu_palette_source_tag` — stable `u32` tag constants for `PaletteSource`
  - `npr::compiled_color::GpuColorProgram` — upload-ready flat `[u32]` bytecode stream with `as_words` / `byte_len` / `deserialize` (round-trip check that decodes the stream back into a `CompiledColorPipeline`)
  - `npr::compiled_color::SerializeError` / `DeserializeError` — non-panicking error surface (`Fallback` rejected up front; unknown opcode / truncated payload / unknown palette source detected on decode)
  - `npr::compiled_color::opcode_word_count` — payload-word count lookup keyed by opcode tag
  - `CompiledColorPipeline::serialize() -> Result<GpuColorProgram, SerializeError>` — encode the CPU-side opcode stream into a GPU-uploadable buffer
  - `npr::compiled_color::emit_wgsl_bytecode_evaluator() -> String` — canonical WGSL source that defines `AliceNprBytecodeCtx` + `alice_npr_eval_bytecode(program_len, ctx) -> vec3<f32>` (stack depth 32). The caller supplies `fn alice_npr_load(index: u32) -> u32`, decoupling the evaluator from any specific bind-group layout and avoiding the `unrestricted_pointer_parameters` WGSL extension.
- Round-trip tests: `serialize_all_native_variants_roundtrip` covers every native opcode; `serialize_deep_composition_roundtrip` covers a nine-level composition tree. Encode → decode → scalar `eval` matches the original pipeline lane-for-lane.
- Naga validation tests (new `tests/npr_bytecode_wgsl_validate.rs`): the emitted evaluator wrapped in a minimal fragment-shader entry point parses (`naga::front::wgsl::parse_str`) and passes full semantic validation (`naga::valid::Validator` with `ValidationFlags::all()`).
- **Phase 12-D** — `CompiledColorPipeline` native opcode coverage extended to all 17 current `NprColorNode` variants. New `ColorOp` variants: `Multiply` / `Add` / `OutlineOver` / `Fresnel` / `Saturate` / `Bloom` / `PosterizeColor` / `Vignette` / `Palette3` / `Palette5` / `Hatch` / `Tonemap` / `SpeedLine`. A well-formed pipeline compiled from any current DSL surface now contains zero `Fallback` opcodes; the `Fallback` opcode is preserved as a forward-compat seam for future variants.
- **Phase 13** — 8-lane SIMD batch evaluator via `wide::f32x8`. New public API:
  - `npr::compiled_color::NprColorBatch8` — SoA 8-lane RGB colour batch with `splat` / `from_vec3s` / `to_vec3s` / `lerp` / `scale` / `mul_componentwise` / `add_vec3x8` / `dot_scalar` / `max_channel`
  - `npr::compiled_color::NprBatchContext8` — SoA 8-lane shading context (derived scalars only: `n_dot_l` / `n_dot_v` / `sdf` / `uv_x` / `uv_y` / `time`), built from `[NprColorContext; 8]` via `from_contexts`
  - `CompiledColorPipeline::eval_batch8(&NprBatchContext8) -> NprColorBatch8` — evaluates the same bytecode across 8 lanes in parallel
  - SIMD-native path for 14 opcodes (`PushConstant` / `Toon` / `SoftToon` / `TwoTone` / `Multiply` / `Add` / `Scale` / `OutlineOver` / `Saturate` / `Bloom` / `PosterizeColor` / `Vignette` / `Palette3` / `Hatch` / `Tonemap`); per-lane scalar over the SoA batch for the remaining 3 (`Fresnel` uses `powf`, `SpeedLine` uses `atan2`, `Palette5` walks a 4-segment palette).
- Benchmarks: `bench_color_pipeline` gains `deep_composition_eval` / `deep_composition_compiled_eval` (6-level composition touching `Toon` + `OutlineOver` + `Fresnel` + `Vignette` + `Saturate` + `Tonemap`) plus P13's `toon_batch8_eval` / `toon_with_outline_batch8_eval` / `deep_composition_batch8_eval` per-call figures.
- Tests: `npr::compiled_color::tests` gains 16 native-opcode coverage tests plus `all_current_variants_compile_without_fallback` regression guard, and 19 batched tests (`vec3x8_from_vec3s_roundtrip` + `batch_matches_scalar_for_*` for every current variant + full-variant composition regression) asserting `eval_batch8` == 8 x `eval` per lane.

### Changed

- `npr::dsl::palette_source_scalar` promoted to `pub(crate)` so `compiled_color::ColorOp::{Palette3, Palette5}` can share the tree-eval scalar-source semantics.
- `prelude` re-exports `NprBatchContext8` and `NprColorBatch8` from `npr::compiled_color`.

### Performance (Apple Silicon, Phase 13)

Per-lane cost of the batched path (total time / 8):

| Pipeline | Tree eval (scalar) | Compiled scalar | Batch8 per-lane | Batch8 vs tree |
|----------|--------------------|-----------------|-----------------|----------------|
| `toon` shallow | 4.98 ns | 21.3 ns | 5.2 ns | ~1.04× (parity) |
| `toon` + outline | ~5.0 ns | 21.7 ns | 5.5 ns | ~1.10× |
| Deep 6-level composition | 18.2 ns | 28.9 ns | **11.4 ns** | **0.63×** |

The deep-composition case is the first regime where the compiled pipeline beats the tree walker outright. Shallow `toon` remains parity because opcode-fetch overhead dominates trivial arithmetic.

## [v1.8.0] - 2026-09-13

**NPR module landing** — a new procedural NPR (Non-Photorealistic Rendering) subsystem across 12 phases (P1 through P12-A) landing as `alice_sdf::npr`. The 5 bridge dependencies (`alice-codec` / `alice-physics` / `alice-cache` / `alice-font` / `alice-asp`) remain trimmed as in 1.7.7 because they have not yet been published to crates.io; scheduled restoration in a future release once upstream publishes.

### Added

- **`npr` module** — Procedural NPR (Non-Photorealistic Rendering) primitives across 9 categories, all closed-form and texture-free (Phase 2 Law-only compliant)
  - `npr::toon` — `toon_ramp`, `soft_toon_ramp`, `two_tone`, `posterize_color`
  - `npr::outline` — `distance_field_outline{,_soft}`, `curvature_outline`, `depth_step_outline`, `composite_outline`
  - `npr::sky` — `sky_gradient_bands`, `puffy_cloud_layer`, `distance_color_quantize`, `light_shaft_beam`, `sun_disc`
  - `npr::rim` — `fresnel_rim`, `procedural_matcap` (2x2 palette bilinear, no texture), `stylized_specular`
  - `npr::hatch` — `hatch_lines`, `cross_hatch`, `paper_grain`, `pencil_shade`
  - `npr::distortion` — `hand_drawn_jitter`, `sketch_wobble`, `line_boil`
  - `npr::palette` — `palette_gradient`, `time_of_day`, `season_palette`
  - `npr::composition` — `vignette`, `bloom_toon`, `chromatic_offsets`
  - `npr::motion` — `speed_line`, `impact_flash`
  - `npr::noise` — `NoiseField` trait + `HashNoise` deterministic hash-based value noise + `PerlinNoise` gradient noise + `WorleyNoise` cellular noise + `SimplexNoise` skewed-lattice gradient noise + `fbm` multi-octave composer
  - `npr::sdf_integration` — Adapters that consume `SdfNode` via `eval`, `eval_normal`, and `autodiff::mean_curvature`: `curvature_outline_from_node`, `distance_outline_from_node`, `toon_shade_from_node`, `soft_toon_shade_from_node`
  - `npr::dsl` — `NprColorNode` expression tree + `NprColorContext` for composing NPR primitives into a color pipeline
  - `npr::shader_glue` — Core (14 primitives) + palette (`sky_gradient_bands_3`, `palette_gradient_5`, `time_of_day`, `season_palette`) GLSL / WGSL / HLSL helper string constants + `helpers_for` / `palette_helpers_for` / `full_helpers_for(ShaderLanguage)` dispatch
- All NPR items re-exported from the `prelude` module
- `examples/npr_toon_demo.rs` — 9-category primitive tour
- `examples/npr_background_scene.rs` — Shadertoy-style raymarching background scene composing multiple NPR primitives
- `benches/npr_primitives.rs` — Criterion benchmarks across all 9 categories plus noise and DSL evaluation
- `npr::scene_composer::SceneShaderBuilder` (feature-gated: `glsl` / `hlsl` / `gpu`) — Builder that composes the NPR helper library, the transpiled SDF evaluator, and a canonical raymarching `main()` per shader language into a single shader source string
  - `.with_pipeline(NprColorNode)` — Replace the default `soft_toon + composite_outline` hit-branch colour block with a custom `NprColorNode` expression tree
- `npr::dsl_shader::transpile_npr_color_node` — Transpile an `NprColorNode` DSL tree into a shader-language snippet (`NprShaderSnippet`) usable across GLSL / WGSL / HLSL
- `NprColorNode` new variants: `Multiply` / `Add` / `Scale` / `Fresnel` / `Saturate` / `Bloom` / `PosterizeColor` / `Vignette` / `Palette3` with builder helpers (`.multiply`, `.plus`, `.scale`, `.with_fresnel`, `.saturate`, `.bloom`, `.posterize`, `.vignetted`)
- `PaletteSource` enum (`NDotL` / `NDotV` / `Sdf` / `UvY`) driving `Palette3`
- `NprColorContext.uv: Vec2` + `NprShaderContext.uv: &str` for UV-dependent variants
- `alice_saturate(color, factor)` added to `NPR_GLSL_HELPERS` / `NPR_WGSL_HELPERS` / `NPR_HLSL_HELPERS`
- `alice_palette_gradient_3(t, c0, c1, c2)` added to `NPR_*_PALETTE_HELPERS`
- `NprColorNode::Hatch { base, angle_rad, density, thickness, ink }` variant with `.with_hatch` builder helper
- `alice_hatch_lines(uv, angle_rad, density, thickness)` added to `NPR_GLSL_HELPERS` / `NPR_WGSL_HELPERS` / `NPR_HLSL_HELPERS`
- `NprColorNode::Palette5 { source, c0..c4 }` variant reusing `alice_palette_gradient_5`
- `NprColorNode::Tonemap { child, exposure }` variant with `.tonemap_reinhard` builder helper
- `NprColorNode::SpeedLine { base, focus, count, thickness, ink }` variant with `.with_speed_lines` builder helper
- `alice_tonemap_reinhard(color, exposure)` and `alice_speed_line(uv, focus, count, thickness)` added to `NPR_GLSL_HELPERS` / `NPR_WGSL_HELPERS` / `NPR_HLSL_HELPERS`
- `NprColorContext.time: f32` + `NprShaderContext.time: &str` (canonical `"iTime"`) for animation
- `PaletteSource::TimeCycle` — `fract(time)` driver for cyclic palettes
- `SceneShaderBuilder` shader output now declares an `iTime` uniform (`layout(binding=0) uniform SceneUniforms.iTime` in GLSL, `SceneUniforms.iTime` in WGSL aliased as `iTime` in `fs_main`, `cbuffer SceneCB.iTime` in HLSL)
- `npr::compiled_color::CompiledColorPipeline` — Host-side bytecode compilation of `NprColorNode` trees into a flat `ColorOp` stream evaluated by a small stack machine. Currently natively supports `Constant` / `Toon` / `SoftToon` / `TwoTone` / `Scale`; other variants use a transparent `Fallback` opcode that delegates to the recursive tree walker. Ships now to lock in the API ahead of SIMD / GPU integration; on shallow trees scalar bytecode is presently slower than tree eval (measured on Apple Silicon: 3.5 ns vs 19 ns for `toon`)
- `benches/npr_primitives.rs::bench_color_pipeline` gains `toon_compiled_eval` and `toon_with_outline_compiled_eval` benchmarks that compare the compiled pipeline against tree evaluation
- `NprShaderContext.n_dot_v` field for Fresnel-driven pipelines; canonical scene shader now declares `ndv = -dot(n, ray_dir)` in the hit branch
- `tests/npr_shader_validate.rs` — Naga-based validation of `SceneShaderBuilder` GLSL and WGSL output (default pipeline + `.with_pipeline` custom trees), plus `naga::valid::Validator` semantic validation on the full-variant WGSL pipeline
- `alice_sun_disc` added to `NPR_GLSL_HELPERS` / `NPR_WGSL_HELPERS` / `NPR_HLSL_HELPERS`
- `examples/npr_scene_shader.rs` — Emit a fully-composed shader for a small CSG scene via `SceneShaderBuilder`
- `.github/workflows/npr-bench.yml` — Benchmark regression watchdog that compares NPR primitive latency between PR head and `main` baseline

## [v1.7.7] - 2026-09-12

**crates.io landing** — first release published to https://crates.io/crates/alice-sdf Absorbs the Unreleased mesh-optimization batch plus the 1.7.4-1.7.6 preparation work (bridge trim + security fixes + fuzz + CI hardening)

### Security

- **RUSTSEC-2025-0020** (pyo3 `PyString::from_object` buffer overflow) — resolved by pyo3 `0.23 → 0.29` major bump
- **RUSTSEC-2026-0177** (pyo3 `PyCFunction::new_closure` `Sync` missing) — resolved by pyo3 `0.23 → 0.29`
- **RUSTSEC-2025-0141** (bincode 1.x unmaintained) — resolved by bincode `1.3 → 2.0`; wire format compat kept via `config::legacy()` so existing `.asdf` files remain readable

### Removed (temporary, restoration scheduled in 1.8.0)

- **5 optional path deps + associated features**: `alice-codec`, `alice-physics`, `libasp` (ALICE-Streaming-Protocol), `alice-cache`, `alice-font` are removed from `[dependencies]`, and the matching features `codec` / `physics` / `asp` / `sdf-cache` / `font` from `[features]` The corresponding `src/*_bridge.rs` modules remain `#[cfg(feature = "...")]`-gated and simply do not compile on crates.io 1.7.7 Users who need the bridges keep using `path` / `git` deps against the sibling repos
- Previously prepared as v1.7.4 (2026-07-23) but that tag/publish was skipped; this release folds the trim + subsequent 1.7.5 / 1.7.6 (internal) hardening into a single crates.io landing

### Added — Mesh optimization batch (23 methods absorbed from zeux/meshoptimizer)

Large batch of mesh-optimization work absorbing 23 methods from the
zeux/meshoptimizer C++ library, adding meshopt binary-compatible codecs,
`EXT_meshopt_compression` glTF integration, vertex filters, triangle
stripification, and Nanite-style meshlet clusters No breaking API changes;
all additions are opt-in

### Added

#### meshopt binary-compatible codec

- **`mesh::meshopt_index_codec`** — indexcodec v1 port (EdgeFIFO +
  VertexFIFO + 16-entry `codeaux` table + 4-mode encoding: edge FIFO
  match, codeaux fast path, full triangle encode, reset detection)
  Public API: `encode_index_buffer(indices) -> Vec<u8>`,
  `decode_index_buffer(bytes, index_count) -> Result<Vec<u32>, CodecError>`
- **`mesh::meshopt_vertex_codec`** — vertexcodec v0/v1 port (16-byte
  groups + bit widths 0/1/2/4/8 + control byte 4-mode: bit-encoded,
  zero, literal, XOR+rotate channel) Public API:
  `encode_vertex_buffer(data, size)`,
  `encode_vertex_buffer_level(data, size, level)` with `level` selecting
  `0=scalar / 2=u8-u16 estimate / 3=u8-u16-u32 XOR+rot estimate`,
  `decode_vertex_buffer(bytes, count, size)`
- **`estimate_rotate` heuristic** — 8-rotation bit-consistency search
  matching meshopt `estimateRotate`, activated at `level >= 3`
- **`tests/meshopt_reference_vectors.rs`** — cross-verification against
  8 fixtures generated by the meshoptimizer v0.24+ C++ library
  (`tri_single`, `strip_small`, `seq_100`, `large_500`, `uniform`),
  proving binary compatibility of the Rust decoder with C++-encoded bytes

#### glTF `EXT_meshopt_compression`

- **`io::meshopt_gltf`** module — compact GLB writer applying meshopt
  encoding to POSITION / NORMAL / TEXCOORD_0 / JOINTS_0 / WEIGHTS_0 /
  indices `MeshoptGltfConfig { export_normals, export_uvs, level,
  double_sided }` with public API `export_glb_meshopt` /
  `export_glb_meshopt_bytes` and skinned variants
  `export_glb_meshopt_skinned` / `export_glb_meshopt_bytes_skinned`
- **`MeshoptSkinning { joints: Vec<[u8; 4]>, weights: Vec<[u8; 4]> }`**
  — external per-vertex bone indices + weights for glTF skinning without
  extending the `Vertex` struct (backward compat)
- **`GltfConfig::meshopt_compress: bool` + `meshopt_level: u8`** — enable
  the meshopt path from the existing `export_glb` / `export_glb_bytes`
  entry points via delegation to `io::meshopt_gltf`; existing
  KHR_mesh_quantization / material / bufferView paths are unaffected
  when the option is off

#### Vertex filters

- **`mesh::meshopt_filter`** module — three encoders + matching in-place
  decoders:
  - **Octahedral** (`encode_filter_oct_i16` / `decode_filter_oct_i16_in_place`)
    — unit-vector projection for normals/tangents, 50–75% smaller than
    raw `f32×3` storage with <1% angular error
  - **Quaternion** (`encode_filter_quat_i16` / `decode_filter_quat_i16_in_place`)
    — largest-component + cyclic-swizzle storage, double-cover discards sign
  - **Exponential** (`encode_filter_exp_u32` / `decode_filter_exp_u32_in_place`)
    — per-lane mantissa (24 bit) + shared exponent (8 bit) pack

#### Mesh optimization primitives

- **`mesh::stripifier`** — Evans-Skiena-Varshney greedy strip generation
  (`stripify(indices, vertex_count, restart_index)` /
  `unstripify(strip, restart_index)`) with 8-triangle lookahead buffer,
  primitive-restart or degenerate-triangle joining Empirical index
  reduction ~48% on closed sphere meshes
- **`mesh::meshlet`** — Nanite-style meshlet clustering
  (`build_meshlets_scan` V1 and `build_meshlets_adjacency` V2 with
  `MeshletConfig::quality()` enabling `adjacency_grow=true` +
  `cone_weight=0.25`) Emits Vulkan `VK_EXT_mesh_shader` /
  DirectX 12 mesh-shader-ready cluster data
- **`ClusterBounds` + `NormalCone { axis, cutoff_cos, apex }`** —
  cluster culling data with `cone_apex` computed via
  `NormalCone::from_normals_and_positions` for tighter backface rejection
- **`mesh::overdraw::optimize_overdraw`** — view-independent triangle
  sort preserving vertex-cache clusters, plus
  `optimize_overdraw_with_views` for custom view directions
- **`mesh::spatial_order::optimize_spatial_order`** — 30-bit Morton
  Z-order spatial locality reorder for BVH build speedup
- **`mesh::optimize::optimize_vertex_fetch`** + **`compute_atvr`** —
  vertex-fetch order + Average Transformed Vertex Ratio metric

#### Quantization + mesh codec

- **`mesh::quantization`** — snorm/unorm i8/i16 encode/decode helpers +
  IEEE 754 binary16 (`half_encode` / `half_decode`)
- **`mesh::mesh_codec`** — custom varint delta codec (independent from
  meshopt binary format) with header `b"ASDF"`, LEB128 varint, zigzag
  signed delta, per-slot triangle index delta encoding, per-byte
  position stream delta Typical index 2–3×, regular position 4–8×
  compression
- **glTF quantization integration** —
  `GltfConfig::quantize_positions` refactored to
  center-based `snorm_i16_encode` (full i16 range, 2× precision vs the
  legacy `[0, 32767]` half-range mapping); new
  `GltfConfig::quantize_normals` (`SBYTE snorm`),
  `quantize_uvs` (`USHORT unorm`), `quantize_colors` (`UBYTE unorm`),
  `quantize_tangents` (`SBYTE snorm`) with unified
  `KHR_mesh_quantization` extension trigger

#### Simplifier / decimation

- **`DecimateConfig::lock_vertices: Vec<bool>`** — per-vertex lock mask
  (meshopt `lockVertices` equivalent) for LOD-seam preservation
- **`DecimateConfig::error_absolute: bool`** (default `true`) — when
  `false`, `max_error` is scaled by the mesh AABB diagonal so the
  threshold applies proportionally to the mesh size, matching the
  meshopt `simplifier.cpp` non-`SimplifyErrorAbsolute` semantics

#### Mesh repair + UV metrics

- **`MeshRepair::orient_faces`** — BFS + signed-volume face reorientation
  for consistent winding
- **`MeshRepair::fill_holes`** — connected-component + triangle-fan hole
  filling
- **`MeshRepair::drop_specks`** — Union-Find + `min_ratio` (`f32`)
  small-island removal
- **`compute_uv_density`** — per-face texel-density measurement with
  `UvDensityReport`, `RECOMMENDED_MIN_TEXELS_PER_FACE = 30.0`, and
  `WARN_LOW_DENSITY_RATIO = 0.05`

#### glTF materials

- **`GltfConfig::double_sided`** — force `doubleSided: true` on all
  materials, addresses back-face culling of mixed-orientation faces from
  Dual Contouring / Marching Cubes output

### References

- zeux/meshoptimizer (MIT) v0.24+ — indexcodec, vertexcodec, stripifier,
  simplifier, clusterizer, overdrawoptimizer, spatialorder,
  vfetchoptimizer, quantization, vertexfilter
- Evans, Skiena, Varshney "Optimizing Triangle Strips for Fast
  Rendering" (1996)
- Cigolle et al "A Survey of Efficient Representations for Independent
  Unit Vectors" (2014)
- Fabian Giesen "Simple lossless index buffer compression" (2013)
- Conor Stokes "Vertex Cache Optimised Index Buffer Compression" (2014)

### Added — Morphology (SDF offset + tolerance fit check for 3-D-print clearance)

- **`morphology` module** (~300 LOC): SDF morphological operations for CAD tolerance / print-clearance workflows
  - `eval_offset(node, point, radius)`: canonical signed offset (exact for `A ⊕ B_r` dilate when `r > 0`, `A ⊖ B_r` erode when `r < 0`)
  - `eval_offset_batch` / `eval_offset_batch_parallel`: batch variants matching the existing `shell` module API surface
  - `tolerance_fits(inner, outer, tolerance, samples, half_extent)`: sample-based test that `inner ⊂ outer ⊕ B_tolerance`
  - `tolerance_max_violation(...)`: worst-case penetration depth (ALICE-Bamboo safety validator uses this for auto-adjusting clearance)
  - Tests: 10 unit (offset scalar/batch/parallel + tolerance-fits accept/reject + violation reporting + panic paths); library total 1311 → 1321 passing
  - Note: set-theoretic `open` / `close` compositions do not reduce to closed-form SDF on arbitrary shapes; flagged as future work in module docs

### Fixed

- **Fuzz-found DoS in `load_asdf`** (`src/io/asdf.rs`): valid ASDF magic + malformed body triggered a bincode 2 `decode_from_slice` `Vec` capacity-overflow panic (attacker-controlled `.asdf` could abort the process). Fixed by `bincode::config::legacy().with_limit::<256 MB>()` allocation cap + panic → `Err(IoError::Serialization)` graceful conversion. Wire format compat preserved (encode side bit-exact) Regression test `test_malformed_body_no_panic` uses the exact fuzz artifact
- **`cargo fmt` regression** (`src/python/*.rs` × 4 sites): `Python::detach(|| ...)` single-line collapse was missed in the pyo3 `0.23 → 0.29` migration; applied and CI restored
- **CI `stub-guard` regex false-positive**: trait default methods with `panic!("... not implemented by this backend")` were being flagged as unshipped stubs; regex tightened to `panic!\([^)]*STUB` (uppercase-only), `todo!` / `unimplemented!` still detected

### Changed — CI hardening

- **`actions/checkout` `@v4 → @v5`** across all workflows (Node.js 20 deprecation), 20+ sites
- **`security-audit.yml` — 2 new informational jobs**: `coverage` (`cargo-llvm-cov`) and `semver-checks` (`cargo-semver-checks`) Semver-checks runs `continue-on-error: true` because 1.8.0 physics/font restoration is a planned major-bump event
- **`security-audit` path filter expanded**: `.github/actions/**` and sibling crate `Cargo.toml` under `examples/*/` and `bindings/**/` now trigger the workflow
- **`alice-stubs` action Cargo.toml template**: `license = "MIT OR Apache-2.0"` added so `cargo-deny` licenses check passes for CI-generated bridge stub crates (stubs are ephemeral, not shipped)
- **`deny.toml`**: `[[licenses.exceptions]]` for `alice-physics` (AGPL-3.0, internal sibling crate) added to bypass mechanical SPDX rejection under `--all-features`; redundant empty `exceptions = []` removed
- **`machete` CI job**: `alice-stubs` step wired for path-dep resolution + `[package.metadata.cargo-machete].ignored` added to 3 Cargo.toml files (`alice-sdf` / `alice-sdf-wasm` / `alice-sdf-bevy`) to suppress false positives for feature-gated planned deps
- **`cargo-fuzz` scaffold + Fuzz workflow** (`.github/workflows/fuzz.yml`) — 3 fuzz targets (`fuzz_sdf_eval` / `fuzz_asdf_decode` / `fuzz_bincode_roundtrip`), nightly toolchain override, matrix parallel, daily `03:00 UTC` schedule, `workflow_dispatch` with `duration_seconds` input Local 5-second smoke: 609k / 52k / 524k executions, 0 crashes each Day-1 real DoS bug catch (see Fixed above) validated the ROI

## [v1.7.4] - 2026-07-23

_Prepared but never tagged / published; the trim plus subsequent 1.7.5 / 1.7.6 (internal) hardening were folded into v1.7.7 (2026-09-12) Preserved below for historical accuracy of the initial trim plan_


### Removed (temporary, restoration scheduled in 1.8.0)

- **5 optional path deps + associated features**: `alice-codec`,
  `alice-physics`, `libasp` (ALICE-Streaming-Protocol), `alice-cache`,
  `alice-font` were removed from `[dependencies]`, and the matching
  features `codec` / `physics` / `asp` / `sdf-cache` / `font` from
  `[features]`, so the crate can be published to crates.io without
  waiting on the transitive dep chain (15+ crates deep). The
  corresponding `src/*_bridge.rs` modules are unchanged and remain
  `#[cfg(feature = "...")]`-gated — they simply never activate on
  crates.io 1.7.4. Users who need the bridges keep using `path`/`git`
  deps against the sibling repos as before.

### Scheduled restoration (1.8.0)

Once `alice-crypto` / `alice-analytics` / `alice-ml` / `alice-db` /
`alice-cache` / `alice-codec` / `alice-physics` / `libasp` /
`alice-font` (and their own transitive deps) reach crates.io, 1.8.0
will restore the 5 features and dep entries with `version =` pins so
`cargo add alice-sdf --features physics` starts working on
downstream consumers.

### Fixed

- Keyword `signed-distance-function` (24 chars) → `distance-field`
  (14 chars) to satisfy the crates.io 20-char limit.
- Description expanded to note the temporary bridge removal.

## [v1.7.3] - 2026-07-04

### Changed

- **`wgpu` dependency: 23 → 24** — GPU features (`gpu`, `volume`, `gpu-mesh`) の内部 wgpu を major bump。API 表面は不変、`Instance::new()` が `wgpu::InstanceDescriptor` を値渡しから参照渡しに変わったため内部 5 箇所 (`src/mesh/gpu_marching_cubes.rs` / `src/compiled/wgsl/gpu_eval.rs`) で `Instance::new(&desc)` に変更。ALICE-TRT v0.8.0 と wgpu version を揃えて **単一 `GpuDevice` を alice-sdf + alice-trt 間で共有可能** に (下流 crate が両方使う場合の VRAM 節約 + wgpu type mismatch 解消)
- **README** — Engine integrations 列挙を `Unreal Engine 5 / 6` に更新 (英語/日本語)

### Added

- **Unreal Engine 6.0 (UE6) support** — `unreal-plugin/AliceSDF.uplugin` の `EngineVersion` を `6.0.0` に bump (UE6-main `f602d4b` time point)。UE5.5+ で導入された最新 RHI API (= `FRHIBatchedShaderParameters` / `FRHIBufferCreateDesc::CreateVertex/CreateIndex` / 4 引数 `SubscribeToPostProcessingPass` / `DispatchComputeShader` / `IMPLEMENT_GLOBAL_SHADER` / `LAYOUT_FIELD` / `FSceneViewExtensionBase` / `GScreenRectangleVertexBuffer` 等) が UE6 にも残存、`UE_DEPRECATED(6.x)` 0 件確認、Build.cs / `.cpp` / `.h` / `.usf` 改変ゼロで論理互換。実機 UE6 Editor build 検証は別途実施推奨

### Backwards compatibility

- Public API 変更なし
- **注**: `--features gpu` (または `volume` / `gpu-mesh`) を有効化する下流 crate は自身の `wgpu` を 24 に揃える必要あり (同 major でないと `wgpu::Device` / `wgpu::Buffer` の型が別種扱い)。ALICE-Metaverse など path dep で追従する crate は自動同期

## [v1.7.2] - 2026-06-08

### Added

- **core clippy-strict CI** — `clippy` job を informational から `-D warnings` 化 (no-default-features + glsl/hlsl/gpu の 2 matrix)。新 lint 混入を即 CI fail で発見
- **Pre-built wheel CI** (`.github/workflows/release-wheels.yml`) — tag push (`v*`) で linux-x86_64 / linux-aarch64 / macos-arm64 / macos-x86_64 / windows-x86_64 の wheel を maturin で abi3-py310 ビルドして Release に attach。1 wheel で Python 3.10–3.13 をカバー
- **REST server smoke test** CI job — `/version` / `/eval` / `/op` / `/mesh` / `/splat` / `/vox` の全 endpoint を curl で叩く
- **WASM build** CI job — `cargo build --target wasm32-unknown-unknown --features wasm` で artifact 生成検証
- **Three.js TypeScript type-check** CI job — `tsc --noEmit` で TypeScript 健全性確認
- **Mobile sample compile** CI job (macOS runner) — iOS は xcodebuild build-for-testing、Android は `gradlew assembleDebug` でリグレッション検出
- **visionOS XCFramework support** — `mobile/packaging/ios/build-xcframework.sh --with-visionos` で `aarch64-apple-visionos` / `aarch64-apple-visionos-sim` slice を追加 (nightly + `-Z build-std`)
- **REST server hardening**:
  - `Authorization: Bearer <ALICE_SDF_TOKEN>` middleware (env が空でなければ全 endpoint で必須化、`/` `/version` は除外)
  - `tower_governor` レート制限 (per-IP、デフォルト 20 RPS / burst 60、`ALICE_SDF_RPS` / `ALICE_SDF_BURST` で上書き可能)
  - `RequestBodyLimitLayer` で 1 MiB JSON body 上限
- **`docs/USAGE.md` / `docs/USAGE_JP.md`** — README から詳細セクション 1675 行を移動
- **`docs/PUBLISH.md`** — crates.io 配布戦略の現状とロードマップを明文化

### Changed

- **STEP / IGES README claim 是正** (`README.md` / `README_JP.md`) — 「Fusion 360 / SolidWorks / OnShape / Rhino / AutoCAD / FreeCAD 互換」を撤回。実態は `POLY_LOOP` + `FACE_OUTER_BOUND` の faceted mesh / Entity 134+136 FEM mesh で、`MANIFOLD_SOLID_BREP` を要求する CAD ツールでは開けない可能性がある旨を明記
- **REST server resolution / size 検証** — 旧 silent `clamp(8, 192)` を `400 Bad Request` に変更 (out-of-range を明示的にエラー返却)
- **README 分割**: 2585 → 913 行 (35%)、JP も同様
- `pyproject.toml`: `requires-python = ">=3.9"` → `">=3.10"` (abi3-py310 と整合)
- `pyproject.toml`: project version 0.1.0 → 1.7.2 (Cargo.toml と同期)

### Fixed

- `src/python/compiled.rs`: 未使用の `source_node` field 削除 (`dead_code` warning 除去)
- `src/io/iges.rs`: `format!()` を str literal に置換 (clippy `useless_format`)
- `src/io/vox.rs`: `cfg.size.min(256).max(1)` → `cfg.size.clamp(1, 256)` (clippy `manual_clamp`)

### Compatibility

- Mobile: iOS / Android **+ visionOS** (XCFramework スクリプトに追加)
- Unreal Engine: 5.7.0 〜 5.7.4 / 5.8.0-preview-1 (変更なし)

## [v1.7.1] - 2026-06-08

### Added

- **REST server endpoint 拡張** (`server/`) — `POST /mesh` (Marching Cubes vertices+normals+indices)、`POST /splat` (3D Gaussian Splats、`format=bytes` で base64 32-byte stream)、`POST /vox` (voxel 配列) を追加。`/version` が全 endpoint を列挙
- **OpenXR `SceneFrame` / `SphereBeacon` / `RayHit` API** (`bindings/openxr/`) — フレーム 1 回分のシーン状態を builder style で組み立て、head/left/right の raycast と手メッシュ→beacon 最小距離をワンメソッドで取得
- **OpenXR `examples/quest_demo.rs`** — Meta Quest 風の 60-frame loop 完全実装サンプル
- **visionOS `makeSDFMeshEntity` / `makeBlobEntity`** — 任意 SDF closure を voxel-fill で評価し RealityKit `ModelEntity` 化 / 2 球 smooth-union を Rust `AliceSDFFramework` 直呼出で blob 生成
- **visionOS `AliceSDFFramework` 統合** — Swift 側 SDF 計算を Rust UniFFI コアにルーティング (`sdfSphere` / `opSmoothUnion` / `sphereBatch` / `aliceSdfVersion`)、`canImport(AliceSDFFramework)` でフォールバック実装も保持

### Changed

- **STEP / IGES export を Marching Cubes 化** (`src/io/step.rs` / `src/io/iges.rs`) — 旧 naive voxel quads を `mesh::sdf_to_mesh` (実 MC アルゴリズム) に置換。res=16 で <100 verts → 数千 verts の品質向上
- **PyO3 を `abi3-py310` に固定** — Python 3.10 / 3.11 / 3.12 / 3.13 を 1 つの `.so` でサポートし、再ビルド不要に
- **CHANGELOG split**: v0.1.0 – v1.3.0 を `CHANGELOG-history.md` に分離 (本ファイルの肥大化対策)
- **CI `clippy-strict` トリガ拡張**: `paths-filter` で `code` (core src/**) 変更時も mobile wrapper の strict clippy を実行 (uniffi-wrapper は alice-sdf core を path dep として再 clippy するため、core の変更が見落とされる設計ミスを修正)
- **CI mobile job**: `cargo test --tests` (debug プロファイル) を追加し、12 公開 wrapper 関数を 26 統合テストで網羅
- README (英日) の Python 節に abi3 / pre-built wheel 説明を追加

### Fixed

- `src/io/vdb.rs`: `VdbError::Io` / `VdbError::InvalidBounds` の missing variant docs (clippy strict 対応)

### Quality

- **Core**: 1,093 tests passing (+10 vs v1.7.0 — STEP/IGES 各 +3 quality tests + Bevy +7 + OpenXR +8 + mobile +22 + server +4)
- **OpenXR**: 4 → 12 tests
- **Bevy**: 4 → 11 tests (normals 単位長 / vertex bounds / annulus / cap planes / plugin build)
- **mobile/uniffi-wrapper**: 4 unit + **26 integration tests** (全 12 公開関数を網羅)
- **server**: 0 → 4 tests
- 全 clippy-strict (`-D warnings`) pass: openxr / mobile / bevy

## [v1.7.0] - 2026-06-06

### Added

#### 3D / Modern rendering

- **3D Gaussian Splatting I/O** (`src/io/splat.rs`) — Inria 3DGS 互換 `.splat` バイナリ (32 bytes/splat: pos + scale + RGBA + compressed quat) の読書き、`sdf_to_splats()` で SDF 表面近傍を Gaussian Splat 化、4 tests
- **MagicaVoxel I/O** (`src/io/vox.rs`) — `.vox` v150 RIFF (SIZE + XYZI chunks) の読書き、`sdf_to_vox()` で SDF を voxelize、4 tests
- **STEP AP203 export** (`src/io/step.rs`) — ISO 10303-21 ASCII Faceted BREP、Fusion 360 / SolidWorks / OnShape / Rhino / FreeCAD 互換、2 tests
- **IGES export** (`src/io/iges.rs`) — IGES ASCII Entity 134 (Node) + 136 (Finite Element) で三角形メッシュ表現、Rhino / AutoCAD 互換、2 tests

#### Web / Mobile / XR

- **WebXR raymarching helpers** (`src/wasm.rs` 拡張) — `raymarch_sphere` / `raymarch_two_spheres_smooth` / `sphere_batch_flat` で VR/AR コントローラ・ハンドメッシュ用 SDF クエリ
- **Three.js / React Three Fiber TypeScript wrapper** (`bindings/threejs/`) — `@alice-sdf/threejs` npm パッケージ、`AliceSDF` クラス + `createSliceTexture()` Three.js helper + `<AliceSDFSlicePlane>` R3F コンポーネント + WebXR 統合例
- **OpenXR native helpers** (`bindings/openxr/`) — `XrPose` 変換 + `raymarch_sphere` + ハンドメッシュバッチ評価、Meta Quest / PC VR / Apple Vision Pro 対応、3 tests
- **visionOS Swift Package** (`mobile/swift-package-visionos/`) — Apple Vision Pro 用 RealityKit ヘルパー (`makeSphereEntity` / `makeBoxEntity`)、`AliceSDFFramework` (XCFramework) を再利用

#### DCC ツール統合

- **Blender Add-on** (`bindings/blender/`) — Blender 4.0 / 4.2 LTS / 4.4+ プラグイン: `.asdf` Import operator + sphere/box/torus 生成 + N-panel UI
- **Houdini Python plugin** (`bindings/houdini/`) — Houdini 20.0 / 20.5 / 21+ 用 Python SOP body + 自動 install.sh (python3.10libs/3.11libs 検出)
- **Maya Python plugin** (`bindings/maya/`) — Autodesk Maya 2024 / 2025 / 2026+ 用 Python module + MFnMesh + `register_menu()`
- **Nuke Python plugin** (`bindings/nuke/`) — Foundry Nuke 15.x / 16.x 用 Python module + Volume export + Slice render
- **Cinema 4D Python plugin** (`bindings/cinema4d/`) — Maxon Cinema 4D 2024 / 2025 / 2026+ 用 Python module + PolygonObject 生成

#### Cloud / Server

- **REST API server** (`server/`) — `axum` 0.7 + `tokio` 1.40、`POST /eval` (primitive 評価) + `POST /op` (operation) 公開、`alicelaw.net/sdf-metaverse` バックエンド向け

### Changed

- README (英日) に Web/VFX/Bevy/Splat/Vox/Blender/Houdini/Maya/Nuke/Cinema 4D/Three.js セクション追加
- DCC ツールの対応バージョンを各 README で **後方互換維持 + 新バージョン明示** (Maya 2024-2026、Houdini 20.0/20.5/21、Nuke 15.x/16.x、Blender 4.0/4.2/4.4)
- `AliceSDF.uplugin`: VersionName 1.6.0 → 1.7.0、Version 3 → 4

### Fixed

- `src/io/vox.rs` / `src/io/iges.rs`: pub struct field の missing docs (clippy strict 対応)
- `src/io/iges.rs`: unused `mut` / 使われない変数を削除
- `src/eval/mod.rs`: `43758.5453` の f32 過剰精度を `43758.547` に修正 (clippy strict `excessive_precision` 対応)

## [v1.6.0] - 2026-06-06

### Added

- **`wasm` feature** — WebAssembly bindings (browser): wasm-bindgen + js-sys。`sdf_sphere` / `sdf_box` / `sdf_torus_w` / `sdf_cylinder_w` / `sdf_plane_w` / 6 op + `render_sphere_slice_rgba` を JavaScript から呼び出し可能。`cargo build --target wasm32-unknown-unknown --features wasm` で動作
- **`openvdb` feature** — OpenVDB Float Grid I/O (Houdini / Maya / Nuke 等の VFX/DCC ツール連携): `bake_dense_grid()` / `bake_to_vdb()` / `load_dense_grid_from_vdb()`。vdb-rs 0.6 ベース。`io::vdb` モジュール、4 tests
- **Bevy plugin** (`bindings/bevy/alice-sdf-bevy/`) — Bevy 0.18 用 ECS 統合: `AliceSdfPlugin` + `SdfShape` Component (Sphere/Box/Torus/Cylinder)、Mesh 自動生成 system、`examples/sphere_demo.rs`、4 tests
- **CI matrix 拡張**: `wasm` / `openvdb` / `bevy` の 3 ジョブ追加、`physics` strict 化 (continue-on-error 削除 + 実 ALICE-Physics clone + 1088 tests カバー)

### Changed

- README (英日) に "Web (WebAssembly) / VFX (OpenVDB) / Bevy エンジン" セクション追加
- `AliceSDF.uplugin`: VersionName 1.5.0 → 1.6.0、Version 2 → 3

### Quality

- 全 CI matrix green: macOS ARM64 / Linux x86_64 / Windows x86_64 + Mobile + Physics strict + wasm + openvdb + bevy + clippy + clippy-strict + fmt
- 全 strict job pass (continue-on-error なし)

## [v1.5.0] - 2026-06-06

### Added

- **Mobile SDK (iOS / Android)** — `mobile/` 配下に [UniFFI](https://mozilla.github.io/uniffi-rs/) ベースの Swift / Kotlin 公開 SDK
  - `mobile/uniffi-wrapper/` — UDL 定義 + Rust ラッパークレート (`sdfSphere` / `sdfBox` / `sdfTorus` / `sdfCylinder` / `sdfPlane` / `sdfRoundedBox` + 6 op + `sphereBatch` + version)
  - `mobile/packaging/ios/build-xcframework.sh` — `AliceSDF.xcframework` (device 44MB + sim fat 88MB) 自動生成
  - `mobile/packaging/android/build-aar.sh` — 4 ABI `libuniffi_alice_sdf.so` (250-400KB) + Kotlin bindings 自動生成
  - `mobile/swift-package/Package.swift` — SwiftPM パッケージ (binaryTarget + Swift bindings 2層)
  - `mobile/samples/ios-swiftui/` — SwiftUI サンプルアプリ (xcodegen + Bridging Header 方式)
  - `mobile/samples/android-compose/` — Jetpack Compose サンプルアプリ (AGP 8.5.2 + Kotlin 2.0)
  - 実機検証: iPhone 17 Pro Simulator (iOS 26.0) + Pixel 6 Emulator (Android 14 / API 34) で iOS と Android 数値完全一致 (sphere d=0.2806, smooth union=0.2056)
- **Rendering metaverse features** — RenderConfig に分光レンダリング / 破壊 / VFX / マイクロ法線 / インテリアマッピング / dual SDF 還元
- **`WgslShader::transpile_material()`** — WithMaterial サブツリーからマテリアル評価関数を WGSL 生成
- **Terrain primitive** — 地形プリミティブ + フルレンダリングパイプライン
- **`examples/sword.lol`** — LOL DSL で記述した剣のサンプル
- **Unreal Engine 5.8 互換性確認** — `AliceSDF.uplugin` に `"EngineVersion": "5.7.0"` 明示、UE 5.7.0 〜 5.7.4 stable + 5.8.0-preview-1 で改修不要を実証 (Shader Parameter API: `FRHIBatchedShaderParameters` + `SetBatchedShaderParameters` + `FRHIBufferCreateDesc::CreateStructured` + `FRHIViewDesc::CreateBufferSRV/UAV`)
- **README リンク**: ALICE SDF Metaverse demo (https://alicelaw.net/sdf-metaverse) + alicelaw.net repo を Related Projects に追加 (英日)

### Changed

- **CI/CD 大規模強化**:
  - `concurrency: cancel-in-progress` で連続 push 時の前 run 自動 cancel
  - `dorny/paths-filter@v3` で README/docs only push の full CI skip
  - `.github/actions/alice-stubs` composite action で dep stub 生成を DRY 化 (60行 × 2 jobs)
  - `mobile` job 新規: iOS 3 target + Android 4 ABI cross-compile + Swift/Kotlin bindings 生成検証 + gpu (Metal) feature ビルド
  - `clippy-strict` job: mobile/uniffi-wrapper のみ `RUSTFLAGS="-Dwarnings"` 厳格
  - `nick-fields/retry@v3`: cargo build 3 リトライ (HTTP/2 framing layer 一過性失敗対策)
  - `CARGO_NET_RETRY=5` + `CARGO_HTTP_MULTIPLEXING=false` env
  - `fmt` job 拡張: core + mobile/uniffi-wrapper 両方
- **Author email**: `Moroya Sakamoto <sakamoro@alicelaw.net>` に統一 (Cargo.toml authors)

### Fixed

- **`optimize.rs`**: 4 パスを値渡し化、未最適化ノードの deep clone 除去 (perf)
- **BVH**: `split_off` 化、mipchain clone 除去、abm `read_to_end` 事前確保 (perf)
- **`check_min_tests`**: 算術エラー修正、cargo 失敗時の安全な skip
- **`ecosystem-tests` schedule**: 削除 (CI では兄弟クレート不在で動作不可)
- **`transpile_material`** ヘルパー重複定義の排除
- **cargo fmt** 差分修正 (CI rustfmt 互換)

### Quality

- **1,379 tests passing** (src/ 1,375 + mobile/uniffi-wrapper 4), 0 failed (+205 from v1.3.0)
- 0 clippy pedantic+nursery warnings (core)
- 0 clippy `-D warnings` (mobile wrapper、strict mode)
- 0 fmt diffs (core + mobile)
- CI matrix: macOS ARM64 + Linux x86_64 + Windows x86_64 + macOS Mobile cross-compile

### Compatibility

| Platform | Status |
|----------|--------|
| Linux x86_64 / aarch64 | 🟢 |
| macOS Apple Silicon / Intel | 🟢 |
| Windows x86_64 | 🟢 |
| **iOS aarch64 / sim** | 🟢 v1.5.0 新規 |
| **Android arm64-v8a / armv7 / x86_64 / x86** | 🟢 v1.5.0 新規 |
| Unreal Engine 5.7.0 〜 5.7.4 (stable) | 🟢 |
| Unreal Engine 5.8.0-preview-1 | 🟢 改修不要見込み |
