//
//  AliceSDFDemo-Bridging-Header.h
//
//  ALICE-SDF Mobile XCFramework の C-ABI ヘッダを Swift 側に bridge する。
//  UniFFI が生成する alice_sdfFFI.modulemap が Swift compiler から自動認識
//  されない問題の回避策として、Bridging Header 経由で C ヘッダを直接 import。
//

#import "alice_sdfFFI.h"
