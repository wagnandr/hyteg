#include "tinyhhg_core/tinyhhg.hpp"

#include "core/mpi/all.h"
#include "core/debug/all.h"

using namespace hhg;

using walberla::real_t;

void checkComm(std::string meshfile,const uint_t maxLevel, bool bufferComm = false){

  //MeshInfo meshInfo = MeshInfo::fromGmshFile("../../data/meshes/quad_4el.msh");
  MeshInfo meshInfo = MeshInfo::fromGmshFile(meshfile);
  SetupPrimitiveStorage setupStorage(meshInfo, uint_c(walberla::mpi::MPIManager::instance()->numProcesses()));
  std::shared_ptr<PrimitiveStorage> storage = std::make_shared<PrimitiveStorage>(setupStorage);


  const uint_t minLevel = 2;
  //const uint_t maxLevel = 4;
  hhg::DGFunction< uint_t > x("x", storage, minLevel, maxLevel);
  if(bufferComm) {
    x.getCommunicator(maxLevel).get()->setLocalCommunicationMode(communication::BufferedCommunicator::BUFFERED_MPI);
  }

  size_t num = 1;
  x.enumerate(maxLevel,num);

  uint_t numberOfChecks = 0;
  uint_t totalExpectedChecks = 0;

  for(auto &edgeIt : storage->getEdges()){
    if(edgeIt.second.get()->getNumHigherDimNeighbors() == 1){

      totalExpectedChecks += 4;
    } else if(edgeIt.second.get()->getNumHigherDimNeighbors() == 2){
      totalExpectedChecks += 8;
    } else {
      WALBERLA_CHECK(false);
    }
  }

  //// check vertex to edge comm ////
  for (auto &edgeIt : storage->getEdges()) {
    Edge &edge = *edgeIt.second;
    //BubbleEdge::printFunctionMemory(edge,x.getEdgeDataID(),maxLevel);
    uint_t *edgeData = edge.getData(x.getEdgeDataID())->getPointer(maxLevel);
    std::vector<PrimitiveID> nbrVertices;
    edge.getNeighborVertices(nbrVertices);
    for(auto& vertexIt : nbrVertices){
      Vertex* vertex = storage->getVertex(vertexIt.getID());
      uint_t* vertexData = vertex->getData(x.getVertexDataID())->getPointer(maxLevel);
      uint_t vPerEdge = levelinfo::num_microvertices_per_edge(maxLevel);
      uint_t pos;
      if(edge.vertex_index(vertex->getID()) == 0){
        pos = 0;
      } else if(edge.vertex_index(vertex->getID()) == 1) {
        pos = vPerEdge - 2;
      } else {
        WALBERLA_CHECK(false, "vertex not on Edge");
      }
      uint_t index = BubbleEdge::edge_index(maxLevel, pos, stencilDirection::CELL_GRAY_SE );
      WALBERLA_CHECK_UNEQUAL(0,edgeData[index]);
      WALBERLA_CHECK_EQUAL(edgeData[index],
                           vertexData[vertex->face_index(edge.neighborFaces()[0]) * 2]);
      index = BubbleEdge::edge_index(maxLevel,
                                     pos == 0 ? pos + 1 : pos,
                                     stencilDirection::CELL_BLUE_SE );
      WALBERLA_CHECK_UNEQUAL(0,edgeData[index]);
      WALBERLA_CHECK_EQUAL(edgeData[index],
                           vertexData[vertex->face_index(edge.neighborFaces()[0]) * 2 + 1]);
      numberOfChecks += 2;
      if(edge.getNumNeighborFaces() == 2){
        index = BubbleEdge::edge_index(maxLevel, pos, stencilDirection::CELL_GRAY_NE );
        WALBERLA_CHECK_UNEQUAL(0,edgeData[index]);
        WALBERLA_CHECK_EQUAL(edgeData[index],
                             vertexData[vertex->face_index(edge.neighborFaces()[1]) * 2]);
        index = BubbleEdge::edge_index(maxLevel,
                                       pos == 0 ? pos + 1 : pos,
                                       stencilDirection::CELL_BLUE_NW );
        WALBERLA_CHECK_UNEQUAL(0,edgeData[index]);
        WALBERLA_CHECK_EQUAL(edgeData[index],
                             vertexData[vertex->face_index(edge.neighborFaces()[1]) * 2 + 1]);
        numberOfChecks += 2;
      }
    }
  }

  WALBERLA_CHECK_EQUAL(totalExpectedChecks,numberOfChecks);


  /// check face edge comms ///
  numberOfChecks = 0;
  totalExpectedChecks = (2 * hhg::levelinfo::num_microvertices_per_edge(maxLevel) - 3) * 3 * storage->getNumberOfLocalFaces();

  for (auto &faceIt : storage->getFaces()) {
    Face &face = *faceIt.second;
    uint_t *faceData = face.getData(x.getFaceDataID())->getPointer(maxLevel);
    std::vector<PrimitiveID> nbrEdges;
    face.getNeighborEdges(nbrEdges);
    for(uint_t i = 0; i < nbrEdges.size(); ++i){
      Edge* edge = storage->getEdge(nbrEdges[0].getID());
      uint_t* edgeData = edge->getData(x.getEdgeDataID())->getPointer(maxLevel);
      uint_t idxCounter = 0;
      uint_t faceIdOnEdge = edge->face_index(face.getID());
//////////////////// GRAY CELL //////////////////////
      idxCounter = 0;
      auto it = BubbleFace::indexIterator(face.edge_index(edge->getID()),
                                          face.edge_orientation[face.edge_index(edge->getID())],
                                          BubbleFace::CELL_GRAY,
                                          maxLevel);
      for(; it != BubbleFace::indexIterator(); ++it){
        if(faceIdOnEdge == 0) {
          WALBERLA_CHECK_UNEQUAL(0,faceData[*it]);
          WALBERLA_CHECK_EQUAL(edgeData[BubbleEdge::edge_index(maxLevel, idxCounter, stencilDirection::CELL_GRAY_SE)], faceData[*it]);
          numberOfChecks++;
        } else if(faceIdOnEdge == 1){
          WALBERLA_CHECK_UNEQUAL(0,faceData[*it]);
          WALBERLA_CHECK_EQUAL(edgeData[BubbleEdge::edge_index(maxLevel, idxCounter, stencilDirection::CELL_GRAY_NE)], faceData[*it]);
          numberOfChecks++;
        } else{
          WALBERLA_CHECK(false);
        }
        idxCounter++;
      }
//////////////////// BLUE CELL //////////////////////
      idxCounter = 0;
      it = BubbleFace::indexIterator(face.edge_index(edge->getID()),
                                     face.edge_orientation[face.edge_index(edge->getID())],
                                     BubbleFace::CELL_BLUE,
                                     maxLevel);
      for(; it != BubbleFace::indexIterator(); ++it){
        if(faceIdOnEdge == 0) {
          WALBERLA_CHECK_EQUAL(edgeData[BubbleEdge::edge_index(maxLevel, idxCounter + 1, stencilDirection::CELL_BLUE_SE)], faceData[*it]);
          numberOfChecks++;
        } else if(faceIdOnEdge == 1){
          WALBERLA_CHECK_EQUAL(edgeData[BubbleEdge::edge_index(maxLevel, idxCounter + 1, stencilDirection::CELL_BLUE_NW)], faceData[*it]);
          numberOfChecks++;
        } else{
          WALBERLA_CHECK(false);
        }
        idxCounter++;
      }
    }
  }
  WALBERLA_CHECK_EQUAL(totalExpectedChecks,numberOfChecks);

}

int main (int argc, char ** argv )
{

  walberla::mpi::Environment MPIenv( argc, argv);
  walberla::MPIManager::instance()->useWorldComm();
  walberla::debug::enterTestMode();

  checkComm("../../data/meshes/quad_4el.msh",4,true);

  checkComm("../../data/meshes/quad_4el.msh",5,true);

  checkComm("../../data/meshes/quad_4el.msh",4,false);

  checkComm("../../data/meshes/quad_4el.msh",5,false);

  checkComm("../../data/meshes/bfs_12el.msh",3,true);

  checkComm("../../data/meshes/bfs_12el.msh",3,false);

  return 0;



}