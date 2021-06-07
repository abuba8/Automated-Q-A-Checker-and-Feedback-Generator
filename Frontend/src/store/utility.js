export const update = (old, newprop) => {
    return {
        ...old,
        ...newprop
    }
}