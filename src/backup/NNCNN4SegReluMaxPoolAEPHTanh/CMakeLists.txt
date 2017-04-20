
include_directories(
  model
)

add_executable(NNCNN4SegReluMaxPoolAEPHTanh NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegReluMaxPoolAEPHTanh ${LIBS})
