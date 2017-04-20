
include_directories(
  model
)

add_executable(NNCNN4SegAEP NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegAEP ${LIBS})
