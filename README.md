# rustyvantagetree

`rustyvantagetree` is a Rust implementation of a Vantage Point Tree (VP-Tree) for efficient similarity search. It supports customizable distance metrics, tree pruning strategies, and parallelized search operations. The library is designed to handle high-dimensional data and provides a flexible framework for building and querying VP-Trees.

## Features

- **Customizable Distance Metrics**: Includes a Hamming distance implementation and allows for custom metrics.
- **Tree Pruning**: Supports exact and approximate pruning strategies for efficient search.
- **Parallelized Search(\*)**: Leverages the `rayon` library for concurrent search operations.
- **Generics Support**: Designed to work with generic data types and metrics.
- **Top K Elements Search**: Efficiently retrieves the top K closest elements based on a given metric.
- **Benchmarks**: Includes benchmarking for distance calculations and tree operations.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
rustyvantagetree = { path = "." }
```

## Usage

To use the library, create a Vantage Point Tree, add elements, and perform searches. Refer to the examples in the `src/main.rs` file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
