
include_directories(
  model
)

add_executable(NNCNN4SegReluMaxPoolA NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegReluMaxPoolA ${LIBS})
