
include_directories(
  model
)

add_executable(NNCNN4Seg NNCNNLabeler.cpp)
target_link_libraries(NNCNN4Seg ${LIBS})
