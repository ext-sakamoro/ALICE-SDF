// Reference writer for tests/test_alembic_oracle.rs: writes a mesh with the Alembic library (not
// this crate), so the test can compare what the library writes for the same mesh with what this
// crate writes, through tests/oracle/alembic_reader.cpp
//
// Input (text, one item per line, floats as hex bits):
//   meta <key> <value>
//   P <count> <bits...>
//   faceIndices <count> <ints...>
//   faceCounts <count> <ints...>
//   N <count> <bits...>      (optional, per-vertex)
//   uv <count> <bits...>     (optional, per-vertex)
//
// usage: alembic_refwriter <input.txt> <out.abc>
//
// Author: Moroya Sakamoto

#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace Alembic::AbcGeom;

static float from_bits(const std::string& h) {
    unsigned u = static_cast<unsigned>(std::stoul(h, nullptr, 16));
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: alembic_refwriter <input.txt> <out.abc>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    Alembic::AbcCoreAbstract::MetaData md;
    std::vector<float> P, N, uv;
    std::vector<int32_t> fi, fc;
    std::set<std::string> seen;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string tag;
        ls >> tag;
        if (tag == "meta") {
            std::string k, v;
            ls >> k >> v;
            md.set(k, v);
            continue;
        }
        size_t n = 0;
        ls >> n;
        seen.insert(tag);
        for (size_t i = 0; i < n; ++i) {
            std::string t;
            ls >> t;
            if (tag == "P") P.push_back(from_bits(t));
            else if (tag == "N") N.push_back(from_bits(t));
            else if (tag == "uv") uv.push_back(from_bits(t));
            else if (tag == "faceIndices") fi.push_back(std::stoi(t));
            else if (tag == "faceCounts") fc.push_back(std::stoi(t));
        }
    }
    // a mesh with no vertices is allowed (its lines carry a count of 0); a missing line is not
    if (!seen.count("P") || !seen.count("faceIndices") || !seen.count("faceCounts")) {
        std::fprintf(stderr, "alembic_refwriter: no mesh in %s\n", argv[1]);
        return 1;
    }
    OArchive archive(Alembic::AbcCoreOgawa::WriteArchive(), argv[2], md);
    OPolyMesh mesh(OObject(archive, kTop), "mesh");
    ON3fGeomParam::Sample ns;
    OV2fGeomParam::Sample us;
    if (!N.empty()) ns = ON3fGeomParam::Sample(N3fArraySample((const N3f*)N.data(), N.size() / 3), kVertexScope);
    if (!uv.empty()) us = OV2fGeomParam::Sample(V2fArraySample((const V2f*)uv.data(), uv.size() / 2), kVertexScope);
    OPolyMeshSchema::Sample sample(V3fArraySample((const V3f*)P.data(), P.size() / 3),
                                   Int32ArraySample(fi.data(), fi.size()),
                                   Int32ArraySample(fc.data(), fc.size()), us, ns);
    mesh.getSchema().set(sample);
    return 0;
}
