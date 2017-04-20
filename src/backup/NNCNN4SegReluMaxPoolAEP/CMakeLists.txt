
include_directories(
  model
)

add_executable(NNCNN4SegReluMaxPoolAEP NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegReluMaxPoolAEP ${LIBS})
