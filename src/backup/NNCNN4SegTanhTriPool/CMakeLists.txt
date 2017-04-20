
include_directories(
  model
)

add_executable(NNCNN4SegTanhTriPool NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegTanhTriPool ${LIBS})
