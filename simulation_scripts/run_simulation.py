from realtime_decoding.decoding import dummy_decoder

def main():
    syndrome = [0, 1, 0, 1]
    result = dummy_decoder(syndrome)
    print("Decoder result:", result)

if __name__ == "__main__":
    main()