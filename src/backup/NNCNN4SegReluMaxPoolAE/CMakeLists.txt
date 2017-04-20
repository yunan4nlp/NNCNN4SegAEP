
include_directories(
  model
)

add_executable(NNCNN4SegReluMaxPoolAE NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegReluMaxPoolAE ${LIBS})
