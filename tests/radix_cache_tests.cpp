#include "kvvmm/radix_cache.hpp"

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <vector>

namespace {

struct DummyPage final {
    std::uint64_t id;
};

kvvmm::BlockSignature signature(std::initializer_list<std::int32_t> tokens)
{
    const std::vector<std::int32_t> block(tokens);
    return kvvmm::BlockSignature::fromTokens(std::span<const std::int32_t>(block.data(), block.size()));
}

} // namespace

int main()
{
    kvvmm::RadixCache<DummyPage> cache;

    std::vector<kvvmm::BlockSignature> prefix = {
        signature({ 1, 2, 3, 4 }),
        signature({ 5, 6, 7, 8 }),
        signature({ 9, 10, 11, 12 }),
    };

    std::vector<std::shared_ptr<DummyPage>> pages = {
        std::make_shared<DummyPage>(DummyPage{ 10 }),
        std::make_shared<DummyPage>(DummyPage{ 20 }),
        std::make_shared<DummyPage>(DummyPage{ 30 }),
    };

    cache.insertPrefix(prefix, pages);

    {
        const auto exact = cache.lookupPrefix(prefix);
        assert(exact.matchedBlocks() == 3);
        assert(exact.pages[0]->id == 10);
        assert(exact.pages[2]->id == 30);
    }

    std::vector<kvvmm::BlockSignature> partial = {
        prefix[0],
        prefix[1],
        signature({ 99, 100, 101, 102 }),
    };

    {
        const auto partialHit = cache.lookupPrefix(partial);
        assert(partialHit.matchedBlocks() == 2);
        assert(partialHit.pages[1]->id == 20);
    }

    pages.clear();
    const auto expired = cache.lookupPrefix(prefix);
    assert(expired.matchedBlocks() == 0);
    assert(cache.pruneExpired() == 3);
    assert(cache.stats().nodes == 1);

    return 0;
}
