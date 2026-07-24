#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kvvmm {

struct BlockSignature final {
    std::vector<std::int32_t> tokens;
    std::uint64_t low{0};
    std::uint64_t high{0};

    [[nodiscard]] static BlockSignature fromTokens(std::span<const std::int32_t> tokenBlock);
    [[nodiscard]] std::string shortName() const;

    friend bool operator==(const BlockSignature& lhs, const BlockSignature& rhs)
    {
        return lhs.low == rhs.low && lhs.high == rhs.high && lhs.tokens == rhs.tokens;
    }
};

struct BlockSignatureHash final {
    [[nodiscard]] std::size_t operator()(const BlockSignature& value) const noexcept;
};

struct RadixCacheStats final {
    std::size_t nodes{1};
    std::size_t residentPages{0};
};

template <typename PageT>
class RadixCache final {
public:
    struct LookupResult final {
        std::vector<std::shared_ptr<PageT>> pages;

        [[nodiscard]] std::size_t matchedBlocks() const noexcept { return pages.size(); }
    };

    void insertPrefix(std::span<const BlockSignature> blocks, std::span<const std::shared_ptr<PageT>> pages)
    {
        if (blocks.size() != pages.size()) {
            throw std::invalid_argument("RadixCache::insertPrefix requires the same number of blocks and pages.");
        }

        Node* node = &root_;
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            auto [it, inserted] = node->children.try_emplace(blocks[i], nullptr);
            if (inserted) {
                it->second = std::make_unique<Node>();
                ++stats_.nodes;
            }
            node = it->second.get();
            node->page = pages[i];
            node->signature = blocks[i];
        }
        refreshResidentPageCount();
    }

    [[nodiscard]] LookupResult lookupPrefix(std::span<const BlockSignature> blocks) const
    {
        LookupResult result;
        const Node* node = &root_;

        for (const BlockSignature& block : blocks) {
            const auto it = node->children.find(block);
            if (it == node->children.end()) {
                break;
            }

            node = it->second.get();
            std::shared_ptr<PageT> page = node->page.lock();
            if (!page) {
                break;
            }
            result.pages.push_back(std::move(page));
        }

        return result;
    }

    std::size_t pruneExpired()
    {
        std::size_t removed = 0;
        pruneExpired(root_, removed);
        stats_.nodes -= removed;
        refreshResidentPageCount();
        return removed;
    }

    [[nodiscard]] RadixCacheStats stats() const noexcept { return stats_; }

private:
    struct Node final {
        std::unordered_map<BlockSignature, std::unique_ptr<Node>, BlockSignatureHash> children;
        std::weak_ptr<PageT> page;
        BlockSignature signature;
    };

    void refreshResidentPageCount()
    {
        std::size_t count = 0;
        refreshResidentPageCount(root_, count);
        stats_.residentPages = count;
    }

    static void refreshResidentPageCount(const Node& node, std::size_t& count)
    {
        if (!node.page.expired()) {
            ++count;
        }
        for (const auto& child : node.children) {
            refreshResidentPageCount(*child.second, count);
        }
    }

    static bool pruneExpired(Node& node, std::size_t& removed)
    {
        for (auto it = node.children.begin(); it != node.children.end();) {
            if (pruneExpired(*it->second, removed)) {
                it = node.children.erase(it);
                ++removed;
            } else {
                ++it;
            }
        }

        return node.children.empty() && node.page.expired() && !node.signature.tokens.empty();
    }

    Node root_;
    RadixCacheStats stats_;
};

} // namespace kvvmm
