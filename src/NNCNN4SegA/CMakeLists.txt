
include_directories(
  model
)

add_executable(NNCNN4SegA NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegA ${LIBS})
