
include_directories(
  model
)

add_executable(NNCNN4SegTanhTriPoolEX-V2 NNCNNLabeler.cpp)
target_link_libraries(NNCNN4SegTanhTriPoolEX-V2 ${LIBS})
