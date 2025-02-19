#include "utils.h"

// Utility to strip leading and trailing spaces
std::string trim(const std::string &str)
{
  size_t start = str.find_first_not_of(" \t\n\r");
  size_t end = str.find_last_not_of(" \t\n\r");
  return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

// Utility to parse a JSON array into an Armadillo vector
arma::vec parse_json_array(const std::string &json_array)
{
  arma::vec result;
  std::stringstream ss(json_array.substr(1, json_array.size() - 2)); // Remove square brackets
  std::string item;
  
  while (std::getline(ss, item, ','))
  {
    result.insert_rows(result.n_elem, 1);              // Expand the vector
    result(result.n_elem - 1) = std::stod(trim(item)); // Convert to double
  }
  return result;
}


std::string json_safe_number(double value)
{
  if (std::isnan(value))
  {
    return "null"; // JSON-friendly representation for NaN
  }
  else if (std::abs(value) < 1e-12)
  {
    return ("0");
  }
  else
  {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return oss.str();
    //    return std::to_string(value);
  }
}

std::string parse_json_block(const std::string &first_line,
                             std::istream &input_stream)
{
  
  std::string line = first_line;         // Start with the first line
  std::string block = first_line + "\n"; // Include the first line in the block
  int brace_depth = 0;
  
  if (line.find("null") != std::string::npos && line.find("{") == std::string::npos)
  {
    return ""; // Return empty string
  }
  
  // Validate the first line and initialize brace depth
  if (line.find("{") != std::string::npos)
  {
    brace_depth++;
  }
  else
  {
    throw std::invalid_argument("block should contain opening braces in first line");
  }
  
  while (std::getline(input_stream, line))
  {
    line = trim(line); // Trim spaces or newline characters
    
    // Update brace depth
    if (line.find("{") != std::string::npos)
    {
      brace_depth++; // Increase brace depth
    }
    if (line.find("}") != std::string::npos)
    {
      brace_depth--; // Decrease brace depth
    }
    
    block += line + "\n"; // Append the current line
    
    // Stop collecting when all braces are balanced
    if (brace_depth == 0)
    {
      break;
    }
  }
  
  return block;
}