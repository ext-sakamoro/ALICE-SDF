// Independent reader for tests/test_alembic_oracle.rs: reads a .abc with the Alembic library
// (not this crate) and prints what it holds, one fact per line, so the Rust test compares it with
// the mesh bit for bit
//
//   archive_valid 1
//   meta alice:meaning_id <value>            (every archive metadata key)
//   polymesh <full name>
//   P <count> <hex bits of each float>
//   faceIndices <count> <each index>
//   faceCounts <count> <each count>
//   N <count> <hex bits of each float>      (when present)
//   uv <count> <hex bits of each float>     (when present)
//   digest <property> <stored 32 hex> <computed 32 hex>
//   hash <object> <properties hash> <children hash>   (also for the top object `/`)
//   constant <property> <0|1> samples <n>
//
// The stored digest of every array sample is printed beside the one the library computes from the
// sample's bytes: a reader does not check it, so this is where a wrong digest shows
//
// Exits 1 when the archive does not open or holds no PolyMesh (it compared nothing)
//
// Author: Moroya Sakamoto

#include <Alembic/AbcCoreFactory/IFactory.h>
#include <Alembic/AbcGeom/All.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

using namespace Alembic;
namespace AbcA = Alembic::AbcCoreAbstract;

static unsigned bits(float f) {
    unsigned u;
    std::memcpy(&u, &f, 4);
    return u;
}

static void floats(const char* tag, const float* p, size_t n) {
    std::printf("%s %zu", tag, n);
    for (size_t i = 0; i < n; ++i) std::printf(" %08x", bits(p[i]));
    std::printf("\n");
}

static int meshes = 0;

static void digest(const char* name, Abc::IArrayProperty prop) {
    std::printf("constant %s %d samples %zu\n", name, prop.isConstant() ? 1 : 0, prop.getNumSamples());
    AbcA::ArraySampleKey stored;
    prop.getKey(stored, Abc::ISampleSelector(Abc::index_t(0)));
    AbcA::ArraySamplePtr samp;
    prop.get(samp, Abc::ISampleSelector(Abc::index_t(0)));
    std::printf("digest %s %s %s\n", name, stored.digest.str().c_str(), samp->getKey().digest.str().c_str());
}

static void walk(const Abc::IObject& obj) {
    for (size_t i = 0; i < obj.getNumChildren(); ++i) {
        Abc::IObject child = obj.getChild(i);
        AbcA::Digest ph, ch;
        if (child.getPropertiesHash(ph) && child.getChildrenHash(ch))
            std::printf("hash %s %s %s\n", child.getFullName().c_str(), ph.str().c_str(), ch.str().c_str());
        if (AbcGeom::IPolyMesh::matches(child.getHeader())) {
            AbcGeom::IPolyMesh mesh(child, Abc::kWrapExisting);
            AbcGeom::IPolyMeshSchema& s = mesh.getSchema();
            AbcGeom::IPolyMeshSchema::Sample sample;
            s.get(sample);
            std::printf("polymesh %s\n", child.getFullName().c_str());
            Abc::P3fArraySamplePtr p = sample.getPositions();
            floats("P", reinterpret_cast<const float*>(p->getData()), p->size() * 3);
            Abc::Int32ArraySamplePtr fi = sample.getFaceIndices();
            std::printf("faceIndices %zu", fi->size());
            for (size_t k = 0; k < fi->size(); ++k) std::printf(" %d", (*fi)[k]);
            std::printf("\n");
            Abc::Int32ArraySamplePtr fc = sample.getFaceCounts();
            std::printf("faceCounts %zu", fc->size());
            for (size_t k = 0; k < fc->size(); ++k) std::printf(" %d", (*fc)[k]);
            std::printf("\n");
            digest("P", s.getPositionsProperty());
            digest(".faceIndices", s.getFaceIndicesProperty());
            digest(".faceCounts", s.getFaceCountsProperty());
            AbcGeom::IN3fGeomParam np = s.getNormalsParam();
            if (np.valid()) {
                AbcGeom::IN3fGeomParam::Sample ns = np.getExpandedValue();
                floats("N", reinterpret_cast<const float*>(ns.getVals()->getData()), ns.getVals()->size() * 3);
                digest("N", np.getValueProperty());
            }
            AbcGeom::IV2fGeomParam up = s.getUVsParam();
            if (up.valid()) {
                AbcGeom::IV2fGeomParam::Sample us = up.getExpandedValue();
                floats("uv", reinterpret_cast<const float*>(us.getVals()->getData()), us.getVals()->size() * 2);
                digest("uv", up.getValueProperty());
            }
            ++meshes;
        }
        walk(child);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: alembic_reader <file.abc>\n");
        return 2;
    }
    try {
        AbcCoreFactory::IFactory factory;
        AbcCoreFactory::IFactory::CoreType core;
        Abc::IArchive archive = factory.getArchive(argv[1], core);
        if (!archive.valid() || core != AbcCoreFactory::IFactory::kOgawa) {
            std::fprintf(stderr, "alembic_reader: %s is not an Ogawa archive the library opens\n", argv[1]);
            return 1;
        }
        std::printf("archive_valid 1\n");
        const AbcCoreAbstract::MetaData& md = archive.getPtr()->getMetaData();
        for (AbcCoreAbstract::MetaData::const_iterator it = md.begin(); it != md.end(); ++it)
            std::printf("meta %s %s\n", it->first.c_str(), it->second.c_str());
        AbcA::Digest ph, ch;
        if (archive.getTop().getPropertiesHash(ph) && archive.getTop().getChildrenHash(ch))
            std::printf("hash / %s %s\n", ph.str().c_str(), ch.str().c_str());
        walk(archive.getTop());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "alembic_reader: %s\n", e.what());
        return 1;
    }
    if (meshes == 0) {
        std::fprintf(stderr, "alembic_reader: no PolyMesh (compared nothing)\n");
        return 1;
    }
    return 0;
}
