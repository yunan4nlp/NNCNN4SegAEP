
include_directories(
  model
)

add_executable(NNLSTM4Seg NNRNNLabeler.cpp)
target_link_libraries(NNLSTM4Seg ${LIBS})
