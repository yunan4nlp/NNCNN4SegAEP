
include_directories(
  model
)

add_executable(NNCNN4SegSigmoidMaxPool NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegSigmoidMaxPool ${LIBS})
