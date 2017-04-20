
include_directories(
  model
)

add_executable(NNCNN4SegReluMinPool NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegReluMinPool ${LIBS})
