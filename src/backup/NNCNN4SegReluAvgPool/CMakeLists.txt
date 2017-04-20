
include_directories(
  model
)

add_executable(NNCNN4SegReluAvgPool NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegReluAvgPool ${LIBS})
