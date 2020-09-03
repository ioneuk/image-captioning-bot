import json

import torch

from models.caption import caption_image_beam_search

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptionService():

    def __init__(self, checkpoint_file: str, word_map_file: str) -> None:
        super().__init__()
        checkpoint = torch.load(checkpoint_file, map_location=device)
        self.decoder = checkpoint['decoder']
        self.encoder = checkpoint['encoder']

        with open(word_map_file, 'r') as j:
            self.word_map = json.load(j)
            self.reverse_word_map = {v: k for k, v in self.word_map.items()}

        self.word_idx_to_filter = {self.word_map["<unk>"], self.word_map["<start>"], self.word_map["<end>"],
                                    self.word_map["<pad>"]}

    def caption(self, image, beam_size=3):
        caption, _ = caption_image_beam_search(self.encoder, self.decoder, image, self.word_map, beam_size)
        return " ".join([self.reverse_word_map[word_idx] for word_idx in caption if word_idx not in self.word_idx_to_filter])