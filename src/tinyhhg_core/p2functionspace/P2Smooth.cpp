#include "P2Smooth.hpp"
#include "tinyhhg_core/levelinfo.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFIndexing.hpp"
#include "tinyhhg_core/edgedofspace/EdgeDoFIndexing.hpp"

namespace hhg {
namespace P2 {

namespace vertex {

void smoothGSvertexDoF(Vertex vertex, const PrimitiveDataID<StencilMemory<double>, Vertex> &vertexDoFStencil,
                       const PrimitiveDataID<FunctionMemory<real_t>, Vertex> &dstVertexDoFID,
                       const PrimitiveDataID<StencilMemory<double>, Vertex> &edgeDoFStencil,
                       const PrimitiveDataID<FunctionMemory<real_t>, Vertex> &dstEdgeDoFID,
                       const PrimitiveDataID<FunctionMemory<real_t>, Vertex> &getVertexDoFID, uint_t level) {

}
} /// namespace vertex

namespace edge {

void smoothGSvertexDoF(Edge edge, const PrimitiveDataID<StencilMemory<double>, Edge> &vertexDoFStencil,
                       const PrimitiveDataID<FunctionMemory<real_t>, Edge> &dstVertexDoFID,
                       const PrimitiveDataID<StencilMemory<double>, Edge> &edgeDoFStencil,
                       const PrimitiveDataID<FunctionMemory<real_t>, Edge> &dstEdgeDoFID,
                       const PrimitiveDataID<FunctionMemory<real_t>, Edge> &rhsVertexDoFID, uint_t level) {

}

void smoothGSedgeDoF(Edge edge, const PrimitiveDataID<StencilMemory<double>, Edge> &vertexDoFStencil,
                     const PrimitiveDataID<FunctionMemory<real_t>, Edge> &dstVertexDoFID,
                     const PrimitiveDataID<StencilMemory<double>, Edge> &edgeDoFStencil,
                     const PrimitiveDataID<FunctionMemory<real_t>, Edge> &dstEdgeDoFID,
                     const PrimitiveDataID<FunctionMemory<real_t>, Edge> &rhsEdgeDoFID, uint_t level) {

}
} /// namespace edge

namespace face {

void smoothGSedgeDoF(Face face, const PrimitiveDataID<StencilMemory<double>, Face> &vertexDoFStencil,
                     const PrimitiveDataID<FunctionMemory<real_t>, Face> &dstVertexDoFID,
                     const PrimitiveDataID<StencilMemory<double>, Face> &edgeDoFStencil,
                     const PrimitiveDataID<FunctionMemory<real_t>, Face> &dstEdgeDoFID,
                     const PrimitiveDataID<FunctionMemory<real_t>, Face> &rhsEdgeDoFID, uint_t level) {

}
} /// namespace face

} /// namespace P2
} /// namespace hhg