#include "kvvmm/radix_cache.hpp"

#include <iomanip>
#include <sstream>

namespace kvvmm {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

void mix(std::uint64_t& hash, std::uint64_t value)
{
    for (int byte = 0; byte < 8; ++byte) {
        const auto octet = static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu);
        hash ^= octet;
        hash *= kFnvPrime;
    }
}

} // namespace

BlockSignature BlockSignature::fromTokens(std::span<const std::int32_t> tokenBlock)
{
    BlockSignature signature;
    signature.tokens.assign(tokenBlock.begin(), tokenBlock.end());

    std::uint64_t low = kFnvOffset;
    std::uint64_t high = kFnvOffset ^ 0x9e3779b97f4a7c15ull;

    mix(low, tokenBlock.size());
    mix(high, tokenBlock.size() ^ 0xd1b54a32d192ed03ull);

    for (std::int32_t token : tokenBlock) {
        const auto value = static_cast<std::uint32_t>(token);
        mix(low, value);
        mix(high, (static_cast<std::uint64_t>(value) << 32) ^ 0xa0761d6478bd642full);
    }

    signature.low = low;
    signature.high = high;
    return signature;
}

std::string BlockSignature::shortName() const
{
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(8) << static_cast<std::uint32_t>(high >> 32)
        << '-' << std::setw(8) << static_cast<std::uint32_t>(low >> 32);
    return out.str();
}

std::size_t BlockSignatureHash::operator()(const BlockSignature& value) const noexcept
{
    std::uint64_t combined = value.low ^ (value.high + 0x9e3779b97f4a7c15ull + (value.low << 6) + (value.low >> 2));
    combined ^= static_cast<std::uint64_t>(value.tokens.size()) * 0xbf58476d1ce4e5b9ull;
    return static_cast<std::size_t>(combined ^ (combined >> 32));
}

} // namespace kvvmm
