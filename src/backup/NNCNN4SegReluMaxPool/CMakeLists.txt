
include_directories(
  model
)

add_executable(NNCNN4SegReluMaxPool NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegReluMaxPool ${LIBS})
