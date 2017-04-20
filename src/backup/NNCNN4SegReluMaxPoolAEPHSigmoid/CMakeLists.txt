
include_directories(
  model
)

add_executable(NNCNN4SegReluMaxPoolAEPHSigmoid NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegReluMaxPoolAEPHSigmoid ${LIBS})
