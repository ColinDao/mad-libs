import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    """
    Perform masked word prediction and visualize attention scores.
    """

    # Prompt user for text with masked word
    # Example: "I'm going to the [MASK] today."
    text = input("Text: ")

    # Tokenizer for BERT model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Tokenize input
    inputs = tokenizer(text, return_tensors = "tf")

    # Get mask token index
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)

    # Cancel program if no mask token is found
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Instantiate BERT language model
    model = TFBertForMaskedLM.from_pretrained(MODEL)

    # Feed text input to model and obtain outputs including attention
    result = model(**inputs, output_attentions = True)

    # Get the predicted scores for the masked token position
    mask_token_logits = result.logits[0, mask_token_index]

    # Get top K results for masked word
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()

    # Print original sentence with the masked word substituted
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize model's attentions
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """

    # Extract numpy input ids
    input_ids = [x.numpy() for x in inputs["input_ids"][0]]

    # Locate mask token id, or return nothing
    try:
        return input_ids.index(mask_token_id)
    except ValueError:
        return None


def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """

    # Scale attention scores to a range of [0, 255] inclusive
    shade = round(255 * attention_score.numpy())

    # Return RGB shades
    return (shade, shade, shade)


def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """

    # Enumerate the attention layers
    for i, layer in enumerate(attentions, 1):

        # Enumerate the attention heads
        for j, head in enumerate(layer[0], 1):

            # Generate diagram for each head in each layer
            generate_diagram(
                i,
                j,
                tokens,
                head
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """

    # Store size of tokens array
    n = len(tokens)

    # Image dimensions
    image_size = GRID_SIZE * n + PIXELS_PER_WORD

    # Create a blank RGBA image with a black background
    img = Image.new(
        "RGBA",
        (image_size, image_size),
        "black"
    )

    # Initialize drawing context for the image
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):

        # Create new image object for token columns
        token_image = Image.new(
            "RGBA",
            (image_size, image_size),
            (0, 0, 0, 0)
        )

        # Initialize drawing context for the token image
        token_draw = ImageDraw.Draw(token_image)

        # Draw the token label along the right edge for columns
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill = "white",
            font = FONT
        )

        # Rotate the token label image by 90 degrees for vertical alignment
        token_image = token_image.rotate(90)

        # Paste token image on original image
        img.paste(token_image, mask = token_image)

        # Draw token labels for rows along the left side of the image
        _, _, width, _ = draw.textbbox(
            (0, 0),
            token,
            font = FONT
        )

        # Position the row token label and draw it onto the main image
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill = "white",
            font = FONT
        )

    # Draw attention scores as shaded rectangles for each token pair
    for i in range(n):

        # y-coordinate for the row
        y = PIXELS_PER_WORD + i * GRID_SIZE

        for j in range(n):

            # x-coordinate for the column
            x = PIXELS_PER_WORD + j * GRID_SIZE

            # Color shade based on attention score
            color = get_color_for_attention_score(attention_weights[i][j])

            # Draw the rectangle representing the attention score
            draw.rectangle((
                x,
                y,
                x + GRID_SIZE,
                y + GRID_SIZE
            ), fill = color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
