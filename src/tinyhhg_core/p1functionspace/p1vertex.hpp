#ifndef P1VERTEX_HPP
#define P1VERTEX_HPP

#include <fmt/format.h>

#include "tinyhhg_core/levelinfo.hpp"
#include "tinyhhg_core/p1functionspace/p1memory.hpp"

#include <core/mpi/MPIWrapper.h>

namespace hhg
{

/// P1Vertex namespace for P1 macro-vertex kernels
namespace P1Vertex
{
//FIXME this can be removed after me moved into walberla namespace
using namespace walberla::mpistubs;

/// Allocate memory for P1 macro-vertex including halos
/// \param vertex Reference to Vertex the allocated memory will belong to
/// \param memory_id Index of the \ref data
/// \param minLevel The minimum level allocated
/// \param maxLevel The maximum level allocated
///
/// This function allocates (1 + number of adjacent edges of \p vertex) real_ts for each level within the range of \p minLevel and \p maxLevel.
/// The pointers to these arrays are saved in the Vertex' data vector at index \p memory_id.
inline void allocate(Vertex& vertex, size_t memory_id, size_t minLevel, size_t maxLevel)
{
  vertex.memory.push_back(new VertexP1FunctionMemory());

  for (size_t level = minLevel; level <= maxLevel; ++level)
  {
    P1::getVertexFunctionMemory(vertex, memory_id)->addlevel(level, vertex.edges.size());
  }
}

inline void free(Vertex& vertex, size_t memory_id)
{
  delete vertex.memory[memory_id];
  vertex.memory[memory_id] = nullptr;
}

inline void interpolate(Vertex& vertex, size_t memory_id, std::function<real_t(const hhg::Point3D&)>& expr, size_t level)
{
  P1::getVertexFunctionMemory(vertex, memory_id)->data[level][0] = expr(vertex.coords);
}

inline void assign(Vertex& vertex, const std::vector<real_t>& scalars, const std::vector<size_t>& src_ids, size_t dst_id, size_t level)
{
  real_t tmp = scalars[0] * P1::getVertexFunctionMemory(vertex, src_ids[0])->data[level][0];

  for (size_t i = 1; i < src_ids.size(); ++i)
  {
    tmp += scalars[i] * P1::getVertexFunctionMemory(vertex, src_ids[i])->data[level][0];
  }

  P1::getVertexFunctionMemory(vertex, dst_id)->data[level][0] = tmp;
}

inline void add(Vertex& vertex, const std::vector<real_t>& scalars, const std::vector<size_t>& src_ids, size_t dst_id, size_t level)
{
  real_t tmp = 0.0;

  for (size_t i = 0; i < src_ids.size(); ++i)
  {
    tmp += scalars[i] * P1::getVertexFunctionMemory(vertex, src_ids[i])->data[level][0];
  }

  P1::getVertexFunctionMemory(vertex, dst_id)->data[level][0] += tmp;
}

inline real_t dot(Vertex& vertex, size_t lhs_id, size_t rhs_id, size_t level)
{
  return P1::getVertexFunctionMemory(vertex, lhs_id)->data[level][0] *
      P1::getVertexFunctionMemory(vertex, rhs_id)->data[level][0];
}

inline void apply(Vertex& vertex, size_t opr_id, size_t src_id, size_t dst_id, size_t level, UpdateType update)
{
  auto& opr_data = P1::getVertexStencilMemory(vertex, opr_id)->data[level];
  auto& src = P1::getVertexFunctionMemory(vertex, src_id)->data[level];
  auto& dst = P1::getVertexFunctionMemory(vertex, dst_id)->data[level];

  if (update == Replace) {
    dst[0] = opr_data[0] * src[0];
  }
  else if (update == Add) {
    dst[0] += opr_data[0] * src[0];
  }

  for (size_t i = 0; i < vertex.edges.size(); ++i)
  {
    dst[0] += opr_data[i+1] * src[i+1];
  }

}

inline void smooth_gs(Vertex& vertex, size_t opr_id, size_t f_id, size_t rhs_id, size_t level)
{
  auto& opr_data = P1::getVertexStencilMemory(vertex, opr_id)->data[level];
  auto& dst = P1::getVertexFunctionMemory(vertex, f_id)->data[level];
  auto& rhs = P1::getVertexFunctionMemory(vertex, rhs_id)->data[level];

  dst[0] = rhs[0];

  for (size_t i = 0; i < vertex.edges.size(); ++i)
  {
    dst[0] -= opr_data[i+1] * dst[i+1];
  }

  dst[0] /= opr_data[0];
}

/// Function testing the perfomance and features of the walberla Buffersystem instead of manual mpi comm
//inline void pull_halos_bs(Vertex& vertex, size_t memory_id, size_t level, walberla::mpi::BufferSystem bs) {
//  size_t i = 1;
//  int rk = walberla::mpi::MPIManager::instance()->rank();
//
//  for (Edge* edge: vertex.edges) {
//    if(vertex.rank != rk && edge->rank == rk)
//    {
//      if (edge->vertex_index(vertex) == 0)
//      {
//        bs.sendBuffer(vertex.rank) << edge->data[memory_id][level - 2][1];
//      }
//      else
//      {
//        bs.sendBuffer(vertex.rank) << edge->data[memory_id][level - 2][levelinfo::num_microvertices_per_edge(level) - 2];
//      }
//    }
//    else if (vertex.rank == rk)
//    {
//      if (edge->rank == rk)
//      {
//        if(edge->vertex_index(vertex) == 0)
//        {
//          vertex.data[memory_id][level-2][i] = edge->data[memory_id][level-2][1];
//        }
//        else
//        {
//          vertex.data[memory_id][level-2][i] = edge->data[memory_id][level-2][levelinfo::num_microvertices_per_edge(level) - 2];
//        }
//      }
//      else
//      {
//        MPI_Recv(&vertex.data[memory_id][level-2][i], 1, walberla::MPITrait< real_t >::type(), edge->rank, i, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//      }
//    }
//  }
//}

inline void pull_halos(Vertex& vertex, size_t memory_id, size_t level)
{
  size_t i = 1;
  auto MPIManager = walberla::mpi::MPIManager::instance();
  int rk = MPIManager->rank();
  walberla::mpi::BufferSystem bs ( MPIManager->comm() );


  for (Edge* edge : vertex.edges)
  {
    if (vertex.rank == rk)
    {
      if (edge->rank == rk)
      {
        if(edge->vertex_index(vertex) == 0)
        {
          P1::getVertexFunctionMemory(vertex, memory_id)->data[level][i] = P1::getEdgeFunctionMemory(*edge, memory_id)->data[level][1];
        }
        else
        {
          P1::getVertexFunctionMemory(vertex, memory_id)->data[level][i] = P1::getEdgeFunctionMemory(*edge, memory_id)->data[level][levelinfo::num_microvertices_per_edge(level) - 2];
        }
      }
      else
      {
        MPI_Recv(&P1::getVertexFunctionMemory(vertex, memory_id)->data[level][i], 1, walberla::MPITrait< real_t >::type(), edge->rank, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    else if (edge->rank == rk)
    {
      if(edge->vertex_index(vertex) == 0)
      {
        MPI_Send(&P1::getEdgeFunctionMemory(*edge, memory_id)->data[level][1], 1, walberla::MPITrait< real_t >::type(), vertex.rank, i, MPI_COMM_WORLD);
      }
      else
      {
        MPI_Send(&P1::getEdgeFunctionMemory(*edge, memory_id)->data[level][levelinfo::num_microvertices_per_edge(level) - 2], 1, walberla::MPITrait< real_t >::type(), vertex.rank, i, MPI_COMM_WORLD);
      }
    }

    i += 1;
  }
}

inline void prolongate(Vertex& vertex, size_t memory_id, size_t level)
{
  P1::getVertexFunctionMemory(vertex, memory_id)->data[level+1][0] = P1::getVertexFunctionMemory(vertex, memory_id)->data[level][0];
}

inline void prolongateQuadratic(Vertex& vertex, size_t memory_id, size_t level)
{
  prolongate(vertex, memory_id, level);
}

inline void restrict(Vertex& vertex, size_t memory_id, size_t level)
{
  auto& vertex_data_f = P1::getVertexFunctionMemory(vertex, memory_id)->data[level];
  auto& vertex_data_c = P1::getVertexFunctionMemory(vertex, memory_id)->data[level-1];

  vertex_data_c[0] = vertex_data_f[0];

  size_t i = 1;
  for (Edge* edge : vertex.edges)
  {
    vertex_data_c[0] += 0.5 * vertex_data_f[i];
    i += 1;
  }
}

inline void printmatrix(Vertex& vertex, size_t opr_id, size_t src_id, size_t level)
{
  auto& opr_data = P1::getVertexStencilMemory(vertex, opr_id)->data[level];
  auto& src = P1::getVertexFunctionMemory(vertex, src_id)->data[level];

  fmt::printf("%d\t%d\t%e\n", (size_t)src[0], (size_t)src[0], opr_data[0]);

  for (size_t i = 0; i < vertex.edges.size(); ++i)
  {
    fmt::printf("%d\t%d\t%e\n", (size_t)src[0], (size_t)src[i+1], opr_data[i+1]);
  }
}

}
}

#endif /* P1VERTEX_HPP */
