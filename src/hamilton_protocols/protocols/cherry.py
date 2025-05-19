from hamilton_protocols.utils import alpha_to_index
from adaptyv_lab import Protocol
from rich import print
import pandas as pd

lay_file = "C:\\Users\\Adaptyvbio\\dev\\hamilton-protocols\\src\\hamilton_protocols\\layouts\\cherry-pick.lay"
cherry_pick_file = "C:\\Users\\Adaptyvbio\\Documents\\MAPS\\CHP-003.csv"

mapping = pd.read_csv(cherry_pick_file)
src_plate_names = mapping["Source Plate"].unique()
print(f"Source plates: {src_plate_names}")

protocol = Protocol.from_layout("test", lay_file, simulator_mode=False)

if protocol.deck is None:
    raise ValueError("No deck found in the layout file.")
print(protocol.deck.grid)

tip_rack = protocol.deck.get_tip_rack("E1")
src_plates = protocol.deck.get_plate_stack("F4")[: len(src_plate_names)]
dst_plates = protocol.deck.get_plate_stack("F3")[: len(src_plate_names)][::-1]
src_plate = protocol.deck.get_plate("C3")
dst_plate = protocol.deck.get_plate("C2")

tips = tip_rack.to_list()

protocol.initialize()

for src_plate_name in src_plate_names:
    protocol.grip_get(src_plates.pop())
    protocol.grip_place(src_plate, eject_tool=1)

    print(f"Processing source plate: {src_plate_name}")

    plate_df = mapping[mapping["Source Plate"] == src_plate_name]
    for row in plate_df.iterrows():
        source_well = row[1]["Source Well"]
        dest_well = row[1]["Destination Well"]
        print(f"Source: {source_well} in {src_plate_name} -> Dest: {dest_well}")

        protocol.transfer(
            src_plate[*alpha_to_index(source_well)],
            dst_plate[*alpha_to_index(dest_well)],
            tips.pop(0),
            30,
        )

    protocol.grip_get(src_plate)
    protocol.grip_place(dst_plates.pop())


async for response in protocol.run():
    print(response)
