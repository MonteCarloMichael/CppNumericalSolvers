//
// Created by heuer on 10.04.17.
//

#ifndef PROBLEMOBSERVER_H
#define PROBLEMOBSERVER_H

namespace cppoptlib {
    class ProblemObserver {
    public:
        ProblemObserver() {};

        virtual ~ProblemObserver() {};

        virtual void stepPerformed() {};
    };
}

#endif //PROBLEMOBSERVER_H
