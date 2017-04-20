
include_directories(
  model
)

add_executable(NNCNN4SegTanhMaxPool NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegTanhMaxPool ${LIBS})
