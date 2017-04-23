
include_directories(
  model
)

add_executable(NNCNN4SegP NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegP ${LIBS})
