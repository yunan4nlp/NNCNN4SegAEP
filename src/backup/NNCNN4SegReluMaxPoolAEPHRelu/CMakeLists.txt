
include_directories(
  model
)

add_executable(NNCNN4SegReluMaxPoolAEPHRelu NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegReluMaxPoolAEPHRelu ${LIBS})
