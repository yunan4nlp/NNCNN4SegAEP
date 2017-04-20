
include_directories(
  model
)

add_executable(NNRNN4Seg NNRNNLabeler.cpp)
target_link_libraries(NNRNN4Seg ${LIBS})
