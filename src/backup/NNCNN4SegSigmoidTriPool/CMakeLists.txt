
include_directories(
  model
)

add_executable(NNCNN4SegSigmoidTriPool NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegSigmoidTriPool ${LIBS})
