#pragma once

#include "mesh.hpp"
#include "operator.hpp"
#include "tinyhhg_core/types/pointnd.hpp"
#include "tinyhhg_core/types/flags.hpp"

#include <string>
#include <functional>

namespace hhg
{

class Function
{
public:
  Function(const std::string& name, const PrimitiveStorage& storage, size_t minLevel, size_t maxLevel)
    : functionName_(name), storage_(storage), minLevel_(minLevel), maxLevel_(maxLevel)
  {
  }

  virtual ~Function()
  {
  }
  const std::string &getFunctionName() const { return functionName_; }

  const PrimitiveStorage &getStorage() const { return storage_; }

  uint_t getMinLevel() const { return minLevel_; }

  uint_t getMaxLevel() const { return maxLevel_; }
private:

  const std::string functionName_;
  const PrimitiveStorage& storage_;
  const uint_t minLevel_;
  const uint_t maxLevel_;
};

}
