// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "include/spectreye.h"

int main(int argc, char** argv) {
	if(argc < 2)
		return 0;

	bool debug = false;
	float enc = 0.0;

	SpectreyeReading res;

	if(argc == 3) {
		if(argv[2] == "d" || argv[2] == "-d")
			debug = true;
		else
			enc = std::stod(argv[2]);
	} else if(argc == 4) {
		if(argv[3] == "d" || argv[3] == "-d")
			debug = true;
	}

	std::string path = std::string(argv[1]);

	Spectreye* s = new Spectreye(debug); 

	if(path.find("SHMS") != std::string::npos)
		res = s->GetAngleSHMS(path, enc);
	else
		res = s->GetAngleHMS(path, enc);

	std::cout << Spectreye::DescribeReading(res) << std::endl;

	return 0;
}
