
include_directories(
  model
)

add_executable(NNCNN4SegTanhTriPoolEX NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegTanhTriPoolEX ${LIBS})
