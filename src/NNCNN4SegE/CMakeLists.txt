
include_directories(
  model
)

add_executable(NNCNN4SegE NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegE ${LIBS})
