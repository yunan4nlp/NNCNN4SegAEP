
include_directories(
  model
)

add_executable(NNCNN4SegReluTriPool NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegReluTriPool ${LIBS})
